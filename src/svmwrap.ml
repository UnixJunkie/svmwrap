(* Copyright (C) 2021, Francois Berenger

   Tsuda Laboratory, Tokyo University,
   5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan.

   CLI wrapper on top of libsvm-tools.
   To train and test models using svm-train/predict *)

open Printf

module A = BatArray
module CLI = Minicli.CLI
module Fn = Filename
module FpMol = Molenc.FpMol
module Ht = BatHashtbl
module L = BatList
module LO = Line_oriented
module Log = Dolog.Log
module Opt = BatOption
module PHT = Dokeysto_camltc.Db_camltc.RW
module RNG = BatRandom.State
module S = BatString

module SL = struct
  type t = bool * float (* (label, pred_score) *)
  let create (l, s) =
    (l, s)
  let get_score (_l, s) =
    s
  let get_label (l, _s) =
    l
end

module ROC = Cpm.MakeROC.Make(SL)
module Perfs = Perf.Make(SL)

(* float_of_string doesn't parse the scientific notation ! *)
let robust_float_of_string s =
  try Scanf.sscanf s "%f" (fun x -> x)
  with exn ->
    (Log.fatal "Svmwrap.robust_float_of_string: could not parse: %s" s;
     raise exn)

(* kernels we support *)
type kernel = Linear
            | RBF of float (* gamma *)
            | Sigmoid of float * float (* (gamma, r) *)
            | Polynomial of float * float * int (* (gamma, r, degree) *)

type kernel_choice = Lin_K
                   | RBF_K
                   | Sig_K
                   | Pol_K

let kernel_choice_of_string = function
  | "Lin" -> Lin_K
  | "RBF" -> RBF_K
  | "Sig" -> Sig_K
  | "Pol" -> Pol_K
  | x -> failwith
           (sprintf "Svmwrap.kernel_choice_of_string: unsupported: %s" x)

let svm_train, svm_predict =
  (* names on Linux *)
  ("svm-train", "svm-predict")

let get_pIC50 pairs s =
  if pairs then
    try Scanf.sscanf s "%s@,%f,%s" (fun _name pIC50 _features -> pIC50)
    with exn -> (Log.error "Svmwrap.get_pIC50 pairs: cannot parse: %s" s;
                 raise exn)
  else
    (* the sparse data file format for libsvm starts with the
     * target_float_val as the first field followed by
       idx:val space-separated FP values *)
    try Scanf.sscanf s "%f %s" (fun pIC50 _features -> pIC50)
    with exn -> (Log.error "Svmwrap.get_pIC50: cannot parse: %s" s;
                 raise exn)

let pred_score_of_pred_line l =
  try Scanf.sscanf l "%d %f %f" (fun _label act_p _dec_p -> act_p)
  with exn ->
    (Log.fatal "Svmwrap.pred_score_of_pred_line: cannot parse: %s" l;
     raise exn)

let is_active pairs s =
  S.starts_with s (if pairs then "active" else "+1 ")

(* what to do with the created models *)
type model_command = Restore_from of Utls.filename
                   | Save_into of Utls.filename
                   | Discard

let epsilon_SVR = 3 (* cf. svm-train manpage *)
let c_SVC = 0

(* (0 <= epsilon <= max_i(|y_i|)); according to:
   "Parameter Selection for Linear Support Vector Regression."
   Jui-Yang Hsia and Chih-Jen Lin.
   February 2020. IEEE Transactions on Neural Networks and Learning Systems.
   DOI: 10.1109/TNNLS.2020.2967637
   This value is passed via -p to svm-train *)
let epsilon_bounds (ys: float list): float * float =
  let maxi = L.max (L.rev_map (abs_float) ys) in
  Log.info "SVR epsilon range: [0:%g]" maxi;
  (0.0, maxi)

(* constants to specify kernel for svm-train *)
let int_of_kernel = function
  | Linear -> 0
  | Polynomial (_g, _r, _d) -> 1
  | RBF _g -> 2
  | Sigmoid (_g, _r) -> 3

let string_of_kernel k =
  let k_const = int_of_kernel k in
  match k with
  | Linear -> sprintf "-t %d" k_const
  | RBF g -> sprintf "-t %d -g %g" k_const g
  | Sigmoid (g, r) -> sprintf "-t %d -g %g -r %g" k_const g r
  | Polynomial (g, r, d) -> sprintf "-t %d -g %g -r %g -d %d" k_const g r d

let human_readable_string_of_kernel = function
  | Linear -> "Lin"
  | RBF g -> sprintf "RBF(%g)" g
  | Sigmoid (g, r) -> sprintf "Sig(%g,%g)" g r
  | Polynomial (g, r, d) -> sprintf "Pol(%g,%g,%d)" g r d

let single_train_test verbose pairs cmd c w train test =
  let quiet_command = if verbose then "" else "-q" in
  (* train *)
  let train_fn = Fn.temp_file ~temp_dir:"/tmp" "svmwrap_train_" ".txt" in
  LO.lines_to_file train_fn train;
  let replaced, model_fn =
    (* libsvm places the model in the current working dir... *)
    S.replace ~str:(train_fn ^ ".model") ~sub:"/tmp/" ~by:"" in
  assert(replaced);
  let w_str = if w <> 1.0 then sprintf " -w1 %g" w else "" in
  Utls.run_command ~debug:verbose
    (sprintf "%s %s -c %g%s -s %d %s"
       svm_train quiet_command c w_str c_SVC train_fn);
  (* test *)
  let test_fn = Fn.temp_file ~temp_dir:"/tmp" "svmwrap_test_" ".txt" in
  LO.lines_to_file test_fn test;
  let preds_fn = Fn.temp_file ~temp_dir:"/tmp" "svmwrap_preds_" ".txt" in
  (* compute AUC on test set *)
  Utls.run_command ~debug:verbose
    (* '-b 1' forces probabilist predictions instead of raw scores *)
    (sprintf "%s %s -b 1 %s %s %s"
       svm_predict quiet_command test_fn model_fn preds_fn);
  (* extract true labels *)
  let true_labels = L.map (is_active pairs) test in
  (* extact predicted scores *)
  let pred_lines = LO.lines_of_file preds_fn in
  begin match cmd with
  | Restore_from _ -> assert(false) (* not dealt with here *)
  | Discard ->
     if not verbose then
       L.iter (Sys.remove) [train_fn; test_fn; preds_fn; model_fn]
  | Save_into models_fn ->
     (Utls.run_command (sprintf "echo %s >> %s" model_fn models_fn);
      if not verbose then
        L.iter (Sys.remove) [train_fn; test_fn; preds_fn])
  end;
  match pred_lines with
  | header :: preds ->
     begin
       (if header <> "labels 1 -1" then
          Log.warn "Svmwrap.single_train_test: wrong header in preds_fn: %s"
            header);
       let pred_scores = L.map pred_score_of_pred_line preds in
       L.map SL.create (L.combine true_labels pred_scores)
     end
  | _ -> assert(false)

let single_train_test_regr verbose cmd kernel e c train test =
  let quiet_option = if not verbose then "-q" else "" in
  (* train *)
  let train_fn = Fn.temp_file ~temp_dir:"/tmp" "svmwrap_train_" ".txt" in
  LO.lines_to_file train_fn train;
  let replaced, model_fn =
    (* libsvm places the model in the current working dir... *)
    S.replace ~str:(train_fn ^ ".model") ~sub:"/tmp/" ~by:"" in
  assert(replaced);
  let kernel_str = string_of_kernel kernel in
  Utls.run_command ~debug:verbose
    (sprintf "%s %s -s %d %s -c %g -p %g %s %s"
       svm_train quiet_option epsilon_SVR kernel_str c e train_fn model_fn);
  (* test *)
  let test_fn = Fn.temp_file ~temp_dir:"/tmp" "svmwrap_test_" ".txt" in
  LO.lines_to_file test_fn test;
  let preds_fn = Fn.temp_file ~temp_dir:"/tmp" "svmwrap_preds_" ".txt" in
  (* compute R2 on test set *)
  Utls.run_command ~debug:verbose
    (sprintf "%s %s %s %s %s"
       svm_predict quiet_option test_fn model_fn preds_fn);
  let actual_values = L.map (get_pIC50 false) test in
  let pred_lines = LO.lines_of_file preds_fn in
  let nb_preds = L.length pred_lines in
  let test_card = L.length test in
  Utls.enforce (nb_preds = test_card)
    (sprintf "Svmwrap.single_train_test_regr: |preds|=%d <> |test|=%d"
       nb_preds test_card);
  begin match cmd with
    | Restore_from _ -> assert(false) (* not dealt with here *)
    | Discard ->
      (if not verbose then
         L.iter (Sys.remove) [train_fn; test_fn; preds_fn; model_fn])
    | Save_into out_fn ->
      begin
        Sys.rename model_fn out_fn;
        Log.info "model saved to: %s" out_fn;
        (if not verbose then
           L.iter (Sys.remove) [train_fn; test_fn; preds_fn])
      end
  end;
  let pred_values = L.map robust_float_of_string pred_lines in
  (actual_values, pred_values)

(* libsvm wants first feature index=1 instead of 0 *)
(* FBR: bug in case there are no features *)
let increment_feat_indexes features =
  let buff = Buffer.create 1024 in
  let feat_vals = S.split_on_char ' ' features in
  L.iter (fun feat_val ->
      try Scanf.sscanf feat_val "%d:%d"
            (fun feat value ->
               bprintf buff " %d:%d" (feat + 1) value
            )
      with exn ->
        (Log.fatal "Svmwrap.increment_feat_indexes: cannot parse '%s' in '%s'"
           feat_val features;
         raise exn)
    ) feat_vals;
  (* eprintf "len:%d features:%s\nres:%s\n%!"
     (L.length feat_vals) features res; *)
  Buffer.contents buff

(* libsvm line format:
   '-80.56 1:7 2:5 3:8 4:5 5:5 6:4 7:6 8:3 9:5 10:4 11:2'
   molenc line format:
   'molname,-75.63,[0:4;1:4;2:5;3:2;4:3;5:2;6:2;7:2]' *)
let liblinear_line_to_FpMol index l =
  let feat_vals = S.split_on_char ' ' l in
  match feat_vals with
  | [] | [_] -> failwith ("Svmwrap.liblinear_line_to_FpMol: " ^ l)
  | ic50 :: feat_vals ->
    let name = sprintf "mol%d" index in
    let buff = Buffer.create 1024 in
    L.iteri (fun i feat_val ->
        try Scanf.sscanf feat_val "%d:%d"
              (fun feat value ->
                 bprintf buff (if i = 0 then "[%d:%d" else ";%d:%d")
                   (feat - 1) value)
        with exn ->
          (Log.fatal "Svmwrap.liblinear_line_to_FpMol: cannot parse: %s"
             feat_val;
           raise exn)
      ) feat_vals;
    Buffer.add_char buff ']';
    FpMol.create name index (robust_float_of_string ic50)
      (Buffer.contents buff)

let atom_pairs_line_to_csv line =
  (* Example for classification:
   * "active<NAME>,pIC50,[feat:val;...]" -> "+1 feat:val ..."
   * "<NAME>,pIC50,[feat:val;...]" -> "-1 feat:val ..." *)
  match S.split_on_char ',' line with
  | [_name; pIC50; features] ->
    assert(S.left features 1 = "[" && S.right features 1 = "]");
    let semi_colon_to_space = function | ';' -> " "
                                       | x -> (S.of_char x) in
    let features' =
      S.replace_chars semi_colon_to_space (S.chop ~l:1 ~r:1 features) in
    sprintf "%s%s" pIC50 (increment_feat_indexes features')
  | _ -> failwith ("Svmwrap.atom_pairs_line_to_csv: cannot parse: " ^ line)

(* unit tests for atom_pairs_line_to_csv *)
let () =
  let s2 = atom_pairs_line_to_csv "activeMolecule,1.0,[2:1;5:8;123:1]" in
  Utls.enforce ("1.0 3:1 6:8 124:1" = s2) s2

let pairs_to_csv verbose pairs_fn =
  let tmp_csv_fn =
    Fn.temp_file ~temp_dir:"/tmp" "svmwrap_pairs2csv_" ".csv" in
  (if verbose then Log.info "--pairs -> tmp CSV: %s" tmp_csv_fn);
  LO.lines_to_file tmp_csv_fn (LO.map pairs_fn atom_pairs_line_to_csv);
  tmp_csv_fn

(* find best (e, C) configuration by R2 maximization *)
let best_r2 l =
  L.fold_left (fun
                ((_best_e, _best_c, _best_K, best_r2) as best)
                ((_curr_e, _curr_c, _best_K, curr_r2) as new_best) ->
                if best_r2 >= curr_r2 then
                  best
                else
                  new_best
              ) (0.0, 0.0, Linear, 0.0) l

let log_R2 e c kernel r2 =
  (if r2 < 0.3 then Log.error
   else if r2 < 0.5 then Log.warn
   else Log.info) "(e,C,K,R2) = %g %g %s %.3f"
    e c (human_readable_string_of_kernel kernel) r2

(* return the best parameter configuration (epsilon, C, kernel) found *)
let optimize_regr verbose ncores kernels es cs train test =
  let ecks = L.cartesian_product (L.cartesian_product es cs) kernels in
  let e_c_k_r2s =
    Parany.Parmap.parmap ncores (fun ((e, c), kernel) ->
        let act, preds =
          single_train_test_regr verbose Discard kernel e c train test in
        let r2 = Cpm.RegrStats.r2 act preds in
        log_R2 e c kernel r2;
        (e, c, kernel, r2)
      ) ecks in
  best_r2 e_c_k_r2s

let accumulate_scores x y = match (x, y) with
  | ([], sl2) -> sl2
  | (sl1, []) -> sl1
  | (sl1, sl2) ->
    L.map2 (fun (l1, s1) (l2, s2) ->
        assert(l1 = l2);
        (l1, s1 +. s2)
      ) sl1 sl2
  
let average_scores k sls =
  assert(L.length sls = k);
  let sum = L.fold_left accumulate_scores [] sls in
  L.map (fun (l, s) -> (l, s /. (float k))) sum

let balanced_bag pairs rng lines =
  let acts, decs = L.partition (is_active pairs) lines in
  let n =
    let n_acts = L.length acts in
    let n_decs = L.length decs in
    min n_acts n_decs in
  let acts_a = Utls.array_bootstrap_sample rng n (A.of_list acts) in
  let decs_a = Utls.array_bootstrap_sample rng n (A.of_list decs) in
  let tmp_a = A.concat [acts_a; decs_a] in
  A.shuffle ~state:rng tmp_a; (* randomize selected lines order *)
  A.to_list tmp_a

let train_test ncores verbose pairs cmd rng c w k train test =
  if k <= 1 then
    (* we don't use bagging then *)
    single_train_test verbose pairs cmd c w train test
  else (* k > 1 *)
    let bags = L.init k (fun _ -> balanced_bag pairs rng train) in
    let k_score_labels =
      Parany.Parmap.parmap ncores (fun bag ->
          single_train_test verbose pairs cmd c w bag test
        ) bags in
    average_scores k k_score_labels

let nfolds_train_test ncores verbose pairs cmd rng c w k n dataset =
  assert(n > 1);
  L.flatten
    (L.map (fun (train, test) ->
         train_test ncores verbose pairs cmd rng c w k train test
       ) (Cpm.Utls.cv_folds n dataset))
    
let train_test_maybe_nfolds
      ncores nfolds verbose model_cmd rng c' w' k' train test =
  if nfolds <= 1 then
    train_test ncores verbose false model_cmd rng c' w' k' train test
  else (* nfolds > 1 *)
    nfolds_train_test ncores verbose false model_cmd rng c' w' k' nfolds
      (L.rev_append train test)
  
let perf_plot noplot score_labels c' w' k' auc bed =
  let title_str =
    sprintf "C=%g w=%g k=%d AUC=%.3f BED=%.3f"
      c' w' k' auc bed in
  if not noplot then
    let tmp_scores_fn =
      Fn.temp_file ~temp_dir:"/tmp" "linwrap_optimize_" ".txt" in
    Perfs.evaluate_performance
      None None tmp_scores_fn title_str score_labels;
    Sys.remove tmp_scores_fn

(* return the best parameter configuration found in the parameter
   configs list [cwks]: (best_c, best_w, best_k, best_auc) *)
let optimize ncores verbose noplot nfolds model_cmd rng train test cwks =
  match cwks with
  | [] -> assert(false) (* there should be at least one configuration *)
  | [((c', w'), k')] ->
     let for_auc =
       let score_labels =
         train_test_maybe_nfolds
           ncores nfolds verbose model_cmd rng c' w' k' train test in
       A.of_list score_labels in
     ROC.rank_order_by_score_a for_auc;
     let auc = ROC.fast_auc_a for_auc in
     let bed = ROC.fast_bedroc_auc_a for_auc in
     perf_plot noplot for_auc c' w' k' auc bed;
     (c', w', k', auc)
  | _ ->
     Parany.Parmap.parfold ncores
       (fun ((c', w'), k') ->
         let for_auc =
           let score_labels =
             train_test_maybe_nfolds
               1 nfolds verbose model_cmd rng c' w' k' train test in
           A.of_list score_labels in
         ROC.rank_order_by_score_a for_auc;
         let auc = ROC.fast_auc_a for_auc in
         let bed = ROC.fast_bedroc_auc_a for_auc in
         perf_plot noplot for_auc c' w' k' auc bed;
         (c', w', k', auc))
       (fun
          ((_c, _w, _k, prev_best_auc) as prev)
          ((c', w', k', curr_auc) as curr) ->
         if curr_auc > prev_best_auc then
           (Log.info "c: %g w1: %g k: %d AUC: %.3f" c' w' k' curr_auc;
            curr)
         else
           (Log.warn "c: %g w1: %g k: %d AUC: %.3f" c' w' k' curr_auc;
            prev)
       ) (-1.0, -1.0, -1, 0.5) cwks

(* variables to monitor NLopt optimization progress *)
let nlopt_iter = ref 0
let nlopt_best_r2 = ref 0.0

let nlopt_reset_iter_and_r2 () =
  nlopt_iter := 0;
  nlopt_best_r2 := 0.0

(* we don't have a gradient --> _gradient *)
let nlopt_eval_solution verbose train test params _gradient =
  match A.length params with
  | 2 ->
    let e = params.(0) in
    let c = params.(1) in
    let act, preds =
      single_train_test_regr verbose Discard Linear e c train test in
    let curr_r2 = Cpm.RegrStats.r2 act preds in
    nlopt_best_r2 := max !nlopt_best_r2 curr_r2; (* best R2 seen up to now *)
    (if verbose || !nlopt_iter mod 10 = 0 then
       Log.info "%04d %.3f Lin(e=%g,C=%g)=%.3f"
         !nlopt_iter !nlopt_best_r2 e c curr_r2
    );
    incr nlopt_iter;
    curr_r2
  | 3 ->
    let e = params.(0) in
    let c = params.(1) in
    let g = params.(2) in
    let act, preds =
      single_train_test_regr verbose Discard (RBF g) e c train test in
    let curr_r2 = Cpm.RegrStats.r2 act preds in
    nlopt_best_r2 := max !nlopt_best_r2 curr_r2; (* best R2 seen up to now *)
    (if verbose || !nlopt_iter mod 10 = 0 then
       Log.info "%04d %.3f RBF(e=%g,C=%g,g=%g)=%.3f"
         !nlopt_iter !nlopt_best_r2 e c g curr_r2
    );
    incr nlopt_iter;
    curr_r2
  | _ -> failwith "Svmwrap.nlopt_eval_solution: only Linear or RBF kernel"

let nlopt_optimize_regr verbose max_evals kernel
    (e_min, e_def, e_max)
    (c_min, c_def, c_max)
    (g_min, g_def, g_max) train test =
  match kernel with
  | Linear ->
    let ndims = 2 in (* for the linear kernel: e and C *)
    (* local optimizer that will be passed to the global one *)
    let local = Nlopt.(create sbplx ndims) in (* local optimizer: gradient-free *)
    Nlopt.set_max_objective local
      (nlopt_eval_solution verbose train test);
    (* I don't set parameter bounds on the local optimizer, I guess
     * the global optimizer handles this *)
    (* hard/stupid stop conditions *)
    Nlopt.set_stopval local 1.0; (* max R2 *)
    (* smart stop conditions *)
    Nlopt.set_ftol_abs local 0.0001; (* FBR: might need to be tweaked *)
    let global = Nlopt.(create auglag ndims) in (* global optimizer *)
    Nlopt.set_local_optimizer global local;
    Nlopt.set_max_objective global
      (nlopt_eval_solution verbose train test);
    (* bounds for e and C *)
    Nlopt.set_lower_bounds global [|e_min; c_min|];
    Nlopt.set_upper_bounds global [|e_max; c_max|];
    (* hard/stupid stop conditions *)
    Nlopt.set_stopval global 1.0; (* max R2 *)
    (* max number of single_train_test_regr calls *)
    Nlopt.set_maxeval global max_evals;
    (* not so stupid starting solution *)
    let initial_guess = [|0.0; 1.0|] in
    let stop_cond, params, r2 = Nlopt.optimize global initial_guess in
    Log.info "NLopt optimize global: %s" (Nlopt.string_of_result stop_cond);
    let e, c = params.(0), params.(1) in
    (e, c, Linear, r2)
  | RBF _ ->
    let ndims = 3 in (* for the RBF kernel: (e, C, g) *)
    (* local optimizer that will be passed to the global one *)
    let local = Nlopt.(create sbplx ndims) in (* local optimizer: gradient-free *)
    Nlopt.set_max_objective local
      (nlopt_eval_solution verbose train test);
    (* I don't set parameter bounds on the local optimizer, I guess
     * the global optimizer handles this *)
    (* hard/stupid stop conditions *)
    Nlopt.set_stopval local 1.0; (* max R2 *)
    (* smart stop conditions *)
    Nlopt.set_ftol_abs local 0.0001; (* FBR: might need to be tweaked *)
    let global = Nlopt.(create auglag ndims) in (* global optimizer *)
    Nlopt.set_local_optimizer global local;
    Nlopt.set_max_objective global
      (nlopt_eval_solution verbose train test);
    (* bounds for e and C *)
    Nlopt.set_lower_bounds global [|e_min; c_min; g_min|];
    Nlopt.set_upper_bounds global [|e_max; c_max; g_max|];
    (* hard/stupid stop conditions *)
    Nlopt.set_stopval global 1.0; (* max R2 *)
    (* max number of single_train_test_regr calls *)
    Nlopt.set_maxeval global max_evals;
    (* not so stupid starting solution *)
    let initial_guess = [|e_def; c_def; g_def|] in
    let stop_cond, params, r2 = Nlopt.optimize global initial_guess in
    Log.info "NLopt optimize global: %s" (Nlopt.string_of_result stop_cond);
    let e, c, g = params.(0), params.(1), params.(2) in
    (e, c, RBF g, r2)
  | _ -> failwith "Svmwrap.nlopt_optimize_regr: only Linear or RBF kernel"

(* like optimize_regr, but using NxCV *)
let optimize_regr_nfolds ncores verbose nfolds kernels es cs train =
  let train_tests = Cpm.Utls.cv_folds nfolds train in
  let ecks = L.cartesian_product (L.cartesian_product es cs) kernels in
  let n_configs = L.length ecks in
  let configs_mapper, folds_mapper =
    (* what can parallelize more? *)
    if n_configs > nfolds then
      (Parany.Parmap.parmap ncores, L.map)
    else
      (L.map, Parany.Parmap.parmap ncores) in
  let e_c_k_r2s =
    configs_mapper (fun ((e, c), kernel) ->
        let all_act_preds =
          folds_mapper (fun (train', test') ->
              single_train_test_regr verbose Discard kernel e c train' test'
            ) train_tests in
        let acts, preds =
          let xs, ys = L.split all_act_preds in
          (L.concat xs, L.concat ys) in
        let r2 = Cpm.RegrStats.r2 acts preds in
        log_R2 e c kernel r2;
        (e, c, kernel, r2)
      ) ecks in
  best_r2 e_c_k_r2s

let single_train_test_regr_nfolds verbose nfolds nprocs kernel e c train =
  let train_tests = Cpm.Utls.cv_folds nfolds train in
  let all_act_preds =
    Parany.Parmap.parmap nprocs (fun (train', test') ->
        let acts, preds =
          single_train_test_regr verbose Discard kernel e c train' test' in
        (acts, preds)
      ) train_tests in
  let xs, ys = L.split all_act_preds in
  (L.concat xs, L.concat ys)

(* instance-wise normalization *)
(* WARNING: this function expects the data to already be in svm-train format *)
let normalize_line l =
  let tokens = S.split_on_char ' ' l in
  match tokens with
  | [] -> failwith "Svmwrap.normalize_line: empty line"
  | [_label] -> failwith ("Svmwrap.normalize_line: no features: " ^ l)
  | label :: features ->
    let sum = ref 0 in
    let feat_vals =
      L.rev_map (fun feat_val_str ->
          Scanf.sscanf feat_val_str "%d:%d"
            (fun feat value ->
               sum := !sum + value;
               (feat, value))
        ) features in
    let feat_norm_vals =
      let total = float !sum in
      L.rev_map (fun (feat, value) ->
          (feat, (float value) /. total)
        ) feat_vals in
    let buff = Buffer.create 1024 in
    Buffer.add_string buff label;
    L.iter (fun (feat, norm_val) ->
        Printf.bprintf buff " %d:%g" feat norm_val
      ) feat_norm_vals;
    Buffer.contents buff

(* unit tests for normalize_line *)
let () =
  assert(normalize_line "+1 2:1 5:8 123:1" = "+1 2:0.1 5:0.8 123:0.1");
  assert(normalize_line "-1 2:3 4:7" = "-1 2:0.3 4:0.7")

let prepend_scores_by_names verbose quiet_option test_fn model_fn output_fn =
  let tmp_csv_fn = pairs_to_csv verbose test_fn in
  Utls.run_command ~debug:verbose
    (sprintf "%s %s %s %s %s"
       svm_predict quiet_option tmp_csv_fn model_fn output_fn);
  (* output_fn only holds floats now.
     the following prepend each score by the corresp. molecule name
     to reach the following line format:
     ^mol_name\tscore$ *)
  let tmp_names_fn = Fn.temp_file ~temp_dir:"/tmp" "svmwrap_" ".names" in
  (* extract mol. names: in *.AP files, this is the first field *)
  Utls.run_command ~debug:verbose
    (sprintf "cut -d',' -f1 %s > %s" test_fn tmp_names_fn);
  Utls.run_command ~debug:verbose
    (sprintf "paste %s %s > %s; mv %s %s"
       tmp_names_fn output_fn tmp_csv_fn tmp_csv_fn output_fn);
  (if not verbose then Sys.remove tmp_names_fn)

let prod_predict_regr verbose pairs model_fn test_fn output_fn =
  let quiet_option = if not verbose then "-q" else "" in
  if pairs then
    prepend_scores_by_names verbose quiet_option test_fn model_fn output_fn
  else
    Utls.run_command ~debug:verbose
      (sprintf "%s %s %s %s %s"
         svm_predict quiet_option test_fn model_fn output_fn)

let atom_pairs_line_to_csv do_classification line =
  (* Example for classification:
   * "active<NAME>,pIC50,[feat:val;...]" -> "+1 feat:val ..."
   * "<NAME>,pIC50,[feat:val;...]" -> "-1 feat:val ..." *)
  match S.split_on_char ',' line with
  | [name; pIC50; features] ->
     let label_str =
       if do_classification then
         if S.starts_with name "active" then "+1" else "-1"
       else (* regression *)
         pIC50 in
     assert(S.left features 1 = "[" && S.right features 1 = "]");
     let semi_colon_to_space = function | ';' -> " "
                                        | x -> (S.of_char x) in
     let features' =
       S.replace_chars semi_colon_to_space (S.chop ~l:1 ~r:1 features) in
     sprintf "%s%s" label_str (increment_feat_indexes features')
  | _ -> failwith ("Linwrap.atom_pairs_line_to_csv: cannot parse: " ^ line)
  
let pairs_to_csv verbose do_classification pairs_fn =
  let tmp_csv_fn =
    Fn.temp_file ~temp_dir:"/tmp" "linwrap_pairs2csv_" ".csv" in
  (if verbose then Log.info "--pairs -> tmp CSV: %s" tmp_csv_fn);
  LO.lines_to_file tmp_csv_fn
    (LO.map pairs_fn
       (atom_pairs_line_to_csv do_classification));
  tmp_csv_fn

let get_name_from_AP_line pairs l =
  if pairs then
    try
      let name, _rest = S.split ~by:"," l in
      name
    with exn -> (Log.fatal "cannot parse: %s" l; raise exn)
  else
    "" (* not sure there is one in that case *)

let prod_predict ncores verbose pairs model_fns test_fn output_fn =
  let quiet_command =
    if verbose then ""
    else "2>&1 > /dev/null" in
  let pred_fns =
    Parany.Parmap.parfold ncores
      (fun model_fn ->
        let preds_fn =
          Fn.temp_file ~temp_dir:"/tmp" "linwrap_preds_" ".txt" in
        Log.info "preds_fn: %s" preds_fn;
        let tmp_csv_fn =
          if pairs then
            let do_classification = true in
            pairs_to_csv verbose do_classification test_fn
          else
            test_fn in
        Utls.run_command ~debug:verbose
          (* '-b 1' forces probabilist predictions instead of raw scores *)
          (sprintf "%s -b 1 %s %s %s %s"
             svm_predict tmp_csv_fn model_fn preds_fn quiet_command);
        (* FBR: bug around here *)
        (if pairs && not verbose then Sys.remove tmp_csv_fn);
        preds_fn)
      (fun acc preds_fn -> preds_fn :: acc)
      [] model_fns in
  (* all pred files should have this same number of predictions
     plus a header line *)
  let nb_rows = LO.count test_fn in
  let card = 1 + nb_rows in
  Utls.enforce
    (L.for_all (fun fn -> card = (LO.count fn)) pred_fns)
    "Linwrap.prod_predict: linwrap_preds_*.txt: different number of lines";
  let tmp_pht_fn = Fn.temp_file ~temp_dir:"/tmp" "linwrap_" ".pht" in
  let pht = PHT.create tmp_pht_fn in
  Log.info "Persistent hash table file: %s" tmp_pht_fn;
  let nb_models = L.length pred_fns in
  begin match pred_fns with
  | [] -> assert(false)
  | pred_fn_01 :: other_pred_fns ->
     begin
       (* populate ht *)
       Log.info "gathering %d models..." nb_models;
       LO.iteri pred_fn_01 (fun k line ->
           if k = 0 then
             assert(line = "labels 1 -1") (* check header *)
           else
             let pred_act_p = pred_score_of_pred_line line in
             let k_str = string_of_int k in
             PHT.add pht k_str (Utls.marshal_to_string pred_act_p)
         );
       Sys.remove pred_fn_01;
       (* accumulate *)
       L.iteri (fun i pred_fn ->
           Log.info "done: %d/%d" (i + 1) nb_models;
           LO.iteri pred_fn (fun k line ->
               if k = 0 then
                 assert(line = "labels 1 -1") (* check header *)
               else
                 let pred_act_p = pred_score_of_pred_line line in
                 let k_str = string_of_int k in
                 let prev_v: float =
                   Utls.unmarshal_from_string (PHT.find pht k_str) in
                 PHT.replace pht k_str
                   (Utls.marshal_to_string (pred_act_p +. prev_v))
             );
           Sys.remove pred_fn
         ) other_pred_fns;
       Log.info "done: %d/%d" nb_models nb_models
     end
  end;
  (* write them to output file, averaged *)
  LO.with_out_file output_fn (fun out ->
      for i = 1 to nb_rows do
        let k_str = string_of_int i in
        let sum_preds: float =
          Utls.unmarshal_from_string (PHT.find pht k_str) in
        fprintf out "%g\n" (sum_preds /. (float nb_models))
      done
    );
  PHT.close pht;
  (* PHT.destroy pht; *)
  (* the previous line would raise Failure("invalid operation")
   * so we just remove the file instead *)
  Sys.remove tmp_pht_fn;
  if output_fn <> "/dev/stdout" then
    (* compute AUC *)
    let auc =
      let test_lines = LO.lines_of_file test_fn in
      let names = L.map (get_name_from_AP_line pairs) test_lines in
      let true_labels = L.map (is_active pairs) test_lines in
      let pred_scores =
        L.map (fun l -> Scanf.sscanf l "%f" (fun x -> x))
          (LO.lines_of_file output_fn) in
      let score_labels = L.map SL.create (L.combine true_labels pred_scores) in
      let name_scores = L.combine names pred_scores in
      (* prepend score with mol. name *)
      LO.with_out_file output_fn (fun out ->
          L.iter (fun (name, pred_score) ->
              fprintf out "%s\t%g\n" name pred_score
            ) name_scores
        );
      ROC.auc score_labels in
    Log.info "AUC: %.3f" auc

let count_active_decoys pairs fn =
  let n_total = LO.length fn in
  let filter =
    if pairs then
      (fun s -> BatString.starts_with s "active")
    else
      (fun s -> BatString.starts_with s "+1 ") in
  let n_actives = ref 0 in
  LO.iter fn (fun line ->
      if filter line then
        incr n_actives
    );
  let n_decoys = n_total - !n_actives in
  Log.info "%s: |A|/|D|=%d/%d" fn !n_actives n_decoys;
  (!n_actives, n_decoys)

let decode_w_range pairs maybe_train_fn input_fn maybe_range_str =
  match maybe_range_str with
  | None ->
    begin
      let n_acts, n_decs =
        match maybe_train_fn with
        | Some train_fn -> count_active_decoys pairs train_fn
        | None -> count_active_decoys pairs input_fn in
      Utls.enforce (n_acts <= n_decs)
        (sprintf "Svmwrap.decode_w_range: n_acts (%d) > n_decs (%d)"
           n_acts n_decs);
      let max_weight = (float n_decs) /. (float n_acts) in
      Log.info "max weight: %g" max_weight;
      L.frange 1.0 `To max_weight 10 (* default w range *)
    end
  | Some s ->
    try Scanf.sscanf s "%f:%d:%f" (fun start nsteps stop ->
        L.frange start `To stop nsteps)
    with exn -> (Log.fatal "Svmwrap.decode_w_range: invalid string: %s"  s;
                 raise exn)

let decode_e_range maybe_range_str = match maybe_range_str with
  | None -> None
  | Some s ->
    try
      Scanf.sscanf s "%f:%d:%f" (fun start nsteps stop ->
          Some (L.frange start `To stop nsteps)
        )
    with exn -> (Log.fatal "Svmwrap.decode_e_range: invalid string: %s"  s;
                 raise exn)

let decode_c_range (maybe_range_str: string option): float list =
  match maybe_range_str with
  | None -> (* default C range *)
    [0.001; 0.002; 0.005;
     0.01;  0.02;  0.05;
     0.1;   0.2;   0.5;
     1.;    2.;    5.;
     10.;  20.;   50.]
  | Some range_str ->
    L.map robust_float_of_string
      (S.split_on_char ',' range_str)

let decode_k_range (maybe_range_str: string option): int list =
  match maybe_range_str with
  | None ->
    (* default k range *)
    [1; 2; 5; 10; 20; 50]
  | Some range_str ->
    L.map int_of_string
      (S.split_on_char ',' range_str)

let decode_g_range (maybe_range_str: string option): float list =
  match maybe_range_str with
  | None -> (* default gamma range *)
    [0.00001; 0.00002; 0.00005;
     0.0001;  0.0002;  0.0005;
     0.001;   0.002;   0.005;
     0.01;    0.02;    0.05;
     0.1;     0.2;     0.5;
     1.0;     2.0;     5.0; 10.0]
  | Some range_str ->
    L.map robust_float_of_string
      (S.split_on_char ',' range_str)

let decode_r_range (maybe_range_str: string option): float list =
  match maybe_range_str with
  | None -> failwith "Svmwrap.decode_r_range: no default range"
  | Some range_str ->
    L.map robust_float_of_string
      (S.split_on_char ',' range_str)

let decode_d_range (maybe_range_str: string option): int list =
  match maybe_range_str with
  | None -> failwith "Svmwrap.decode_d_range: no default range"
  | Some range_str ->
    L.map int_of_string
      (S.split_on_char ',' range_str)

(* (0 <= epsilon <= max_i(|y_i|)); according to:
   "Parameter Selection for Linear Support Vector Regression."
   Jui-Yang Hsia and Chih-Jen Lin.
   February 2020. IEEE Transactions on Neural Networks and Learning Systems.
   DOI: 10.1109/TNNLS.2020.2967637
   To optimize a SVR, we need to do the exponential scan of C
   for each epsilon value. *)
let svr_epsilon_range (nsteps: int) (ys: float list): float list =
  let maxi = L.max (L.rev_map (abs_float) ys) in
  Log.info "SVR epsilon range: [0:%g]; nsteps=%d" maxi nsteps;
  L.frange 0.0 `To maxi nsteps

let epsilon_range maybe_epsilon maybe_esteps maybe_es train =
  match (maybe_epsilon, maybe_esteps, maybe_es) with
  | (None, None, Some es) -> es
  | (_, _, Some _) ->
    failwith "Svmwrap.epsilon_range: (e or esteps) and --e-range"
  | (Some _, Some _, None) ->
    failwith "Svmwrap.epsilon_range: both e and esteps"
  | (None, None, None) -> [0.1] (* svm-train's default for -p option *)
  | (Some e, None, None) -> [e]
  | (None, Some nsteps, None) ->
    let train_pIC50s = L.map (get_pIC50 false) train in
    let mini, maxi = L.min_max ~cmp:BatFloat.compare train_pIC50s in
    let avg = L.favg train_pIC50s in
    let std = Utls.stddev train_pIC50s in
    Log.info "(min, avg+/-std, max): %.3f %.3f+/-%.3f %.3f"
      mini avg std maxi;
    svr_epsilon_range nsteps train_pIC50s

let read_IC50s_from_train_fn pairs train_fn =
  LO.map train_fn (get_pIC50 pairs)

let read_IC50s_from_preds_fn pairs preds_fn =
  if pairs then
    LO.map preds_fn
      (fun line -> Scanf.sscanf line "%s@\t%f" (fun _name score -> score))
  else
    LO.map preds_fn robust_float_of_string

(* convert an atom-pair line to svm-train csv format plus applies
 * instance-wise normalization *)
let instance_wise_norm_AP_line l =
  let fp_mol =
    let ignore_index = 0 in
    Molenc.FpMol.parse_one ignore_index l in
  let dep_var = FpMol.get_value fp_mol in
  let fp = FpMol.get_fp fp_mol in
  let sum_values = float (Molenc.Fingerprint.sum_values fp) in
  let buff = Buffer.create 80 in
  bprintf buff "%f" dep_var;
  Molenc.Fingerprint.kv_iter (fun k v ->
      (* libsvm_fst_idx=1 IWN(falue) *)
      bprintf buff " %d:%g" (k + 1) ((float v) /. sum_values)
    ) fp;
  (* DON'T terminate with '\n' the line *)
  Buffer.contents buff

type norm_params = { global_min: int;
                     global_max: int;
                     (* LUT: feature -> (min_val, max_val) *)
                     min_max_ht: (int, (int * int)) Ht.t }

(* for each feature, extract min and max values *)
let extract_norm_params_AP_lines num_features lines =
  let ht = Ht.create num_features in
  let glob_min = ref max_int in
  let glob_max = ref min_int in
  L.iter (fun line ->
      let fp_mol =
        let ignore_index = 0 in
        Molenc.FpMol.parse_one ignore_index line in
      let fp = FpMol.get_fp fp_mol in
      Molenc.Fingerprint.kv_iter (fun k v ->
          if v < !glob_min then glob_min := v;
          if v > !glob_max then glob_max := v;
          try
            let prev_min, prev_max = Ht.find ht k in
            Ht.replace ht k (min prev_min v, max prev_max v)
          with Not_found -> Ht.add ht k (v, v)
        ) fp
  ) lines;
  { global_min = !glob_min; global_max = !glob_max; min_max_ht = ht }

(* scale values in [0:1] *)
let apply_norm_params_AP_line params line =
  let glob_min = params.global_min in
  let glob_max = params.global_max in
  let def_val = (glob_min, glob_max) in
  let ht = params.min_max_ht in
  let fp_mol =
    let ignore_index = 0 in
    Molenc.FpMol.parse_one ignore_index line in
  let dep_var = FpMol.get_value fp_mol in
  let fp = FpMol.get_fp fp_mol in
  let buff = Buffer.create 80 in
  bprintf buff "%f" dep_var;
  Molenc.Fingerprint.kv_iter (fun k v ->
      let mini, maxi = Ht.find_default ht k def_val in
      let scaled =
        if maxi = mini then 0.0
        else float (v - mini) /. float (maxi - mini) in
      (* libsvm_fst_idx=1 scale(value) *)
      bprintf buff " %d:%g" (k + 1) scaled
    ) fp;
  (* DON'T terminate with '\n' the line *)
  Buffer.contents buff

type normalization = IWN (* instance-wise normalization *)
                   | Scaled (* scaled to fall in [0:1] *)
                   | None

let lines_of_file num_features pairs2csv do_classification normalize fn =
  let all_lines = LO.lines_of_file fn in
  if pairs2csv then
    match normalize with
    | Scaled ->
      let norm_params =
        extract_norm_params_AP_lines num_features all_lines in
      L.map (apply_norm_params_AP_line norm_params) all_lines
    | IWN -> L.map instance_wise_norm_AP_line all_lines
    | None -> L.map (atom_pairs_line_to_csv do_classification) all_lines
  else
    match normalize with
    | Scaled -> failwith "Svmwrap.lines_of_file: \
                          not --pairs and Scaled: unsupported"
    | IWN -> L.map normalize_line all_lines
    | None -> all_lines

(* uncompress file if needed
   return (uncompressed_fn, was_compressed) *)
let maybe_uncompress fn =
  if S.ends_with fn ".gz" then
    let plain = S.rchop ~n:3 fn in
    Utls.run_command (sprintf "gunzip -f -k %s" fn);
    (plain, true)
  else
    (fn, false)

let main () =
  Log.(set_log_level INFO);
  Log.color_on ();
  let argc, args = CLI.init () in
  if argc = 1 then
    (eprintf "usage: %s\n  \
              -i <filename>: training set or DB to screen\n  \
              --feats <int>: number of features\n  \
              [-o <filename>]: predictions output file\n  \
              [-np <int>]: ncores\n  \
              [--kernel <string>] choose kernel type {Lin|RBF|Sig|Pol}\n  \
              [-c <float>]: fix C\n  \
              [-e <float>]: epsilon in the loss function of epsilon-SVR;\n  \
              (0 <= epsilon <= max_i(|y_i|))\n  \
              [--nlopt <int>]: use NLopt with MAX_ITER (global optim.)\n  \
              instead of grid-search (recommended: MAX_ITER >= 100)\n  \
              [-g <float>]: fix gamma (for RBF and Sig kernels)\n  \
              [-r <float>]: fix r for the Sig kernel\n  \
              [--iwn]: turn ON instance-wise-normalization\n  \
              [--scale]: turn ON [0:1] scaling (NOT PRODUCTION READY)\n  \
              [--no-plot]: no gnuplot\n  \
              [{-n|--NxCV} <int>]: folds of cross validation\n  \
              [-q]: quiet\n  \
              [-v|--verbose]: equivalent to not specifying -q\n  \
              [--seed <int>]: fix random seed\n  \
              [-p <float>]: training set portion (in [0.0:1.0])\n  \
              [--pairs]: read from .AP files (atom pairs; \
              will offset feat. indexes by 1)\n  \
              [--train <train.liblin>]: training set (overrides -p)\n  \
              [--valid <valid.liblin>]: validation set (overrides -p)\n  \
              [--test <test.liblin>]: test set (overrides -p)\n  \
              [{-l|--load} <filename>]: prod. mode; use trained models\n  \
              [{-s|--save} <filename>]: train. mode; save trained models\n  \
              [-f]: force overwriting existing model file\n  \
              [--scan-c]: scan for best C\n  \
              [--scan-e <int>]: epsilon scan #steps for SVR\n  \
              [--scan-g]: scan for best gamma\n  \
              [--regr]: regression (SVR); also, implied by -e and --scan-e\n  \
              [--e-range <float>:<int>:<float>]: specific range for e\n  \
              (semantic=start:nsteps:stop)\n  \
              [--c-range <float,float,...>] explicit scan range for C \n  \
              (example='0.01,0.02,0.03')\n  \
              [--g-range <float,float,...>] explicit range for gamma \n  \
              (example='0.01,0.02,0.03')\n  \
              [--scan-k]: scan number of bags\n  \
              [--k-range <int,int,...>] explicit scan range for k \n  \
              (example='1,2,3,5,10')\n  \
              [-k <int>]: explicit value for k\n  \
              [--r-range <float,float,...>] explicit range for r \n  \
              (example='0.01,0.02,0.03')\n"
       Sys.argv.(0);
     exit 1);
  if CLI.get_set_bool ["-v"] args then Log.(set_log_level DEBUG);
  let input_fn, was_compressed =
    let input_fn' = CLI.get_string_def ["-i"] args "/dev/null" in
    maybe_uncompress input_fn' in
  let maybe_train_fn = CLI.get_string_opt ["--train"] args in
  let maybe_valid_fn = CLI.get_string_opt ["--valid"] args in
  let maybe_test_fn = CLI.get_string_opt ["--test"] args in
  let output_fn = CLI.get_string_def ["-o"] args "/dev/stdout" in
  let will_save = L.mem "-s" args || L.mem "--save" args in
  let will_load = L.mem "-l" args || L.mem "--load" args in
  let force = CLI.get_set_bool ["-f"] args in
  let pairs = CLI.get_set_bool ["--pairs"] args in
  Utls.enforce (not (will_save && will_load))
    ("Svmwrap.main: cannot load and save at the same time");
  let model_cmd =
    begin match CLI.get_string_opt ["-s"; "--save"] args with
      | Some fn ->
        let () =
          Utls.enforce
            (force || not (Sys.file_exists fn))
            ("Svmwrap: file already exists: " ^ fn) in
        Save_into fn
      | None ->
        begin match CLI.get_string_opt ["-l"; "--load"] args with
          | Some fn -> Restore_from fn
          | None -> Discard
        end
    end in
  let ncores = CLI.get_int_def ["-np"] args 1 in
  let train_p = CLI.get_float_def ["-p"] args 0.8 in
  assert(train_p >= 0.0 && train_p <= 1.0);
  let nfolds = CLI.get_int_def ["-n";"--NxCV"] args 1 in
  let rng = match CLI.get_int_opt ["--seed"] args with
    | None -> BatRandom.State.make_self_init ()
    | Some seed -> BatRandom.State.make [|seed|] in
  let scan_C = CLI.get_set_bool ["--scan-c"] args in
  let scan_g = CLI.get_set_bool ["--scan-g"] args in
  let fixed_c = CLI.get_float_opt ["-c"] args in
  let fixed_g = CLI.get_float_opt ["-g"] args in
  let fixed_r = CLI.get_float_opt ["-r"] args in
  let fixed_d = CLI.get_int_opt ["-d"] args in
  let fixed_k = CLI.get_int_opt ["-k"] args in
  let scan_k = CLI.get_set_bool ["--scan-k"] args in
  let k_range_str = CLI.get_string_opt ["--k-range"] args in
  let e_range_str = CLI.get_string_opt ["--e-range"] args in
  let c_range_str = CLI.get_string_opt ["--c-range"] args in
  let g_range_str = CLI.get_string_opt ["--g-range"] args in
  let r_range_str = CLI.get_string_opt ["--r-range"] args in
  let d_range_str = CLI.get_string_opt ["--d-range"] args in
  let quiet = CLI.get_set_bool ["-q"] args in
  let maybe_nlopt = CLI.get_int_opt ["--nlopt"] args in
  let verbose = (not quiet) || (CLI.get_set_bool ["-v";"--verbose"] args) in
  let normalize =
    match (CLI.get_set_bool ["--iwn"] args,
           CLI.get_set_bool ["--scale"] args) with
    | true, false -> IWN
    | false, true -> Scaled
    | true, true -> failwith "Svmwrap: at most one of {--iwn|--scale}"
    | false, false -> None in
  let maybe_epsilon = CLI.get_float_opt ["-e"] args in
  let maybe_esteps = CLI.get_int_opt ["--scan-e"] args in
  let do_regression =
    CLI.get_set_bool ["--regr"] args ||
    Opt.is_some maybe_epsilon || Opt.is_some maybe_esteps ||
    Opt.is_some e_range_str in
  let do_classification = not do_regression in
  let no_gnuplot = CLI.get_set_bool ["--no-plot"] args in
  let chosen_kernel =
    let default_kernel_str = "Lin" in
    kernel_choice_of_string
      (CLI.get_string_def ["--kernel"] args default_kernel_str) in
  let num_features = CLI.get_int ["--feats"] args in
  (* default value from svm-train documentation *)
  let default_gamma = 1.0 /. (float num_features) in
  Utls.enforce (not (L.mem "-e" args && L.mem "--scan-e" args))
    "Svmwrap: -e and --scan-e are exclusive";
  Utls.enforce (not (L.mem "-c" args && L.mem "--scan-c" args))
    "Svmwrap: -c and --scan-c are exclusive";
  Utls.enforce (not (L.mem "-g" args && L.mem "--scan-g" args))
    "Svmwrap: -g and --scan-g are exclusive";
  CLI.finalize (); (* ----------------------------------------------------- *)
  (* scan C? *)
  let cs = match fixed_c with
    | Some c -> [c]
    | None ->
      if scan_C || BatOption.is_some c_range_str then
        decode_c_range c_range_str
      else (* default value from svm-train documentation *)
        [1.0] in
  (* gamma range is handled very similarly to C range *)
  let gs = match fixed_g with
    | Some g -> [g]
    | None ->
      if scan_g || BatOption.is_some g_range_str then
        decode_g_range g_range_str
      else
        [default_gamma] in
  (* r range; only used by the sigmoid and polynomial kernels *)
  let rs = match fixed_r with
    | Some r -> [r]
    | None ->
      if BatOption.is_some r_range_str then
        decode_r_range r_range_str
      else (* default value from svm-train documentation *)
        [0.0] in
  (* d range; only used by the polynomial kernel *)
  let ds = match fixed_d with
    | Some d -> [d]
    | None ->
      if BatOption.is_some d_range_str then
        decode_d_range d_range_str
      else (* default value from svm-train documentation *)
        [3] in
  (* e-range? *)
  let maybe_es = decode_e_range e_range_str in
  let kernels = match chosen_kernel with
    | Lin_K -> [Linear]
    | RBF_K -> L.map (fun g -> RBF g) gs
    | Sig_K ->
      let grs = L.cartesian_product gs rs in
      L.map (fun (g, r) -> Sigmoid (g, r)) grs
    | Pol_K ->
      let grds = L.cartesian_product (L.cartesian_product gs rs) ds in
      L.map (fun ((g, r), d) -> Polynomial (g, r, d)) grds in
  let ws = [1.0] in (* we don't support scanning w yet *)
  (* scan k? *)
  let ks =
    if scan_k || BatOption.is_some k_range_str then
      decode_k_range k_range_str
    else match fixed_k with
         | Some k -> [k]
         | None -> [1] in
  let cwks = L.cartesian_product (L.cartesian_product cs ws) ks in
  begin match model_cmd with
  | Restore_from models_fn ->
     if do_regression then
      begin
        prod_predict_regr verbose pairs models_fn input_fn output_fn;
        let acts = read_IC50s_from_train_fn pairs input_fn in
        let preds = read_IC50s_from_preds_fn pairs output_fn in
        let r2 = Cpm.RegrStats.r2 acts preds in
        let rmse = Cpm.RegrStats.rmse acts preds in
        let title_str =
          sprintf "N=%d R2=%.3f RMSE=%.3f T=%s"
            (L.length preds) r2 rmse input_fn in
        (if not no_gnuplot then
           Gnuplot.regr_plot title_str acts preds
        )
      end
     else
       let model_fns = LO.lines_of_file models_fn in
       prod_predict ncores verbose pairs model_fns input_fn output_fn
    | Save_into (_)
    | Discard ->
      match maybe_train_fn, maybe_valid_fn, maybe_test_fn with
      | (None, None, None) ->
        begin
          (* randomize lines *)
          let all_lines =
            L.shuffle ~state:rng
              (lines_of_file num_features pairs do_classification normalize input_fn) in
          let nb_lines = L.length all_lines in
          (* partition *)
          let train_card =
            BatFloat.round_to_int (train_p *. (float nb_lines)) in
          let train, test = L.takedrop train_card all_lines in
          if do_regression then
            begin
          let best_e, best_c, best_K, best_r2 =
            match maybe_nlopt with
            | Some max_iter ->
              let e_bounds =
                let dep_vars = L.map (get_pIC50 false) all_lines in
                let mini, maxi = epsilon_bounds dep_vars in
                let default = 0.1 in
                (mini, default, maxi) in
              (* cf. "A Practical Guide to Support Vector Classification"
               * Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin; May 19, 2016
               * for those ranges *)
              let c_bounds = (2.0 ** -.5.0, 1.0, 2.0 ** 15.0) in
              let g_bounds = (2.0 ** -.15.0, default_gamma, 2.0 ** 3.0) in
              let e', c', k', r2' =
                nlopt_optimize_regr
                  verbose max_iter (L.hd kernels) e_bounds c_bounds g_bounds train test in
              (e', c', k', r2')
            | None ->
              let epsilons =
                epsilon_range maybe_epsilon maybe_esteps maybe_es train in
              if nfolds <= 1 then
                optimize_regr verbose ncores kernels epsilons cs train test
              else
                optimize_regr_nfolds
                  ncores verbose nfolds kernels epsilons cs all_lines in
          let actual, preds =
            if nfolds <= 1 then
              single_train_test_regr
                verbose model_cmd best_K best_e best_c train test
            else
              let actual', preds' =
                single_train_test_regr_nfolds
                  verbose nfolds ncores best_K best_e best_c
                  all_lines in
              (actual', preds') in
          (* dump to a .act_pred file  *)
          let act_preds = L.combine actual preds in
          let rmse = Cpm.RegrStats.rmse actual preds in
          LO.with_out_file output_fn (fun out ->
              L.iter (fun (act, pred) ->
                  fprintf out "%f\t%f" act pred
                ) act_preds
            );
          let title_str =
            sprintf "nfolds=%d K=%s e=%g C=%g R2=%.3f RMSE=%.3f T=%s"
              nfolds (human_readable_string_of_kernel best_K)
              best_e best_c best_r2 rmse input_fn in
          Log.info "%s" title_str;
          if not no_gnuplot then
            Gnuplot.regr_plot title_str actual preds
            end
          else (* do classification *)
            let best_c, best_w, best_k, best_auc =
              optimize ncores verbose no_gnuplot nfolds
                model_cmd rng train test cwks in
            Log.info "T=%s nfolds=%d C=%.3f w=%.3f k=%d AUC=%.3f"
              input_fn nfolds best_c best_w best_k best_auc
        end
      | (Some _train_fn, Some _valid_fn, Some _test_fn) ->
        failwith "not implemented yet"
      | _ ->
        failwith "Svmwrap: --train, --valid and --test: \
                  provide all three or none"
  end;
  if was_compressed then
    (* don't keep around the uncompressed version *)
    Sys.remove input_fn

let () = main ()
