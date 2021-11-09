(* Copyright (C) 2021, Francois Berenger

   Tsuda Laboratory,
   Tokyo University,
   5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan.

   CLI wrapper on top of libsvm-tools.
   To train and test models using svm-train/predict *)

open Printf

module A = BatArray
module CLI = Minicli.CLI
module Fn = Filename
module FpMol = Molenc.FpMol
module L = BatList
module Log = Dolog.Log
module LO = Line_oriented
module Opt = BatOption
module RNG = BatRandom.State
module S = BatString

let svm_type = 3 (* epsilon-SVR (cf. svm-train manpage) *)

(* kernels we support; Polynomial is not supported because too many parameters *)
type kernel = Linear
            | RBF of float (* gamma *)
            | Sigmoid of float * float (* (gamma, r) *)

let svm_train, svm_predict =
  (* names on Linux *)
  ("svm-train", "svm-predict")

let get_pIC50 pairs s =
  if pairs then
    try Scanf.sscanf s "%s@,%f,%s" (fun _name pIC50 _features -> pIC50)
    with exn -> (Log.error "Svmwrap.get_pIC50 pairs: cannot parse: %s" s;
                 raise exn)
  else
    (* the sparse data file format for liblinear starts with the
     * target_float_val as the first field followed by
       idx:val space-separated FP values *)
    try Scanf.sscanf s "%f %s" (fun pIC50 _features -> pIC50)
    with exn -> (Log.error "Svmwrap.get_pIC50: cannot parse: %s" s;
                 raise exn)

(* what to do with the created models *)
type model_command = Restore_from of Utls.filename
                   | Save_into of Utls.filename
                   | Discard

let single_train_test_regr verbose cmd e c train test =
  let quiet_option = if not verbose then "-q" else "" in
  (* train *)
  let train_fn = Fn.temp_file ~temp_dir:"/tmp" "svmwrap_train_" ".txt" in
  LO.lines_to_file train_fn train;
  let replaced, model_fn =
    (* liblinear places the model in the current working dir... *)
    S.replace ~str:(train_fn ^ ".model") ~sub:"/tmp/" ~by:"" in
  assert(replaced);
  Utls.run_command ~debug:verbose
    (sprintf "%s %s -s 11 -c %g -p %g %s %s"
       svm_train quiet_option c e train_fn model_fn);
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
    | Discard -> L.iter (Sys.remove) [train_fn; test_fn; preds_fn; model_fn]
    | Save_into out_fn ->
      begin
        Sys.rename model_fn out_fn;
        Log.info "model saved to: %s" out_fn;
        L.iter (Sys.remove) [train_fn; test_fn; preds_fn]
      end
  end;
  let pred_values = L.map float_of_string pred_lines in
  (actual_values, pred_values)

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

(* must be greater than 0 and less than 2^30 *)
let rand_int_bound = (BatInt.pow 2 30) - 1

let bagged_train_test_regr nprocs verbose seed_stream cmd e c k train' test =
  if k = 1 then
    (* be compatible with single_train_test_regr *)
    single_train_test_regr verbose cmd e c train' test
  else
    (* bagging regressors *)
    let a = A.of_list train' in
    let n = A.length a in
    let seeds =
      L.init k (fun _i ->
          RNG.int seed_stream rand_int_bound
        ) in
    let preds =
      Parany.Parmap.parmap nprocs (fun seed ->
          let rng = Random.State.make [|seed|] in
          let train = A.to_list (Utls.array_bootstrap_sample rng n a) in
          let acts, preds = single_train_test_regr verbose cmd e c train test in
          L.combine acts preds
        ) seeds in
    L.split (average_scores k preds)

(* liblinear wants first feature index=1 instead of 0 *)
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

(* liblinear line format:
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
    FpMol.create name index (float_of_string ic50) (Buffer.contents buff)

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
  | _ -> failwith ("Svmwrap.atom_pairs_line_to_csv: cannot parse: " ^ line)

(* unit tests for atom_pairs_line_to_csv *)
let () =
  let s1 = atom_pairs_line_to_csv true "activeMOL,1.0,[2:1;5:8;123:1]" in
  Utls.enforce ("+1 3:1 6:8 124:1" = s1) s1;
  let s2 = atom_pairs_line_to_csv false "activeMolecule,1.0,[2:1;5:8;123:1]" in
  Utls.enforce ("1.0 3:1 6:8 124:1" = s2) s2

let pairs_to_csv verbose do_classification pairs_fn =
  let tmp_csv_fn =
    Fn.temp_file ~temp_dir:"/tmp" "svmwrap_pairs2csv_" ".csv" in
  (if verbose then Log.info "--pairs -> tmp CSV: %s" tmp_csv_fn);
  LO.lines_to_file tmp_csv_fn
    (LO.map pairs_fn
       (atom_pairs_line_to_csv do_classification));
  tmp_csv_fn

(* split a list into n parts (the last one might have less elements) *)
let list_nparts n l =
  let len = L.length l in
  assert(n <= len);
  let m = int_of_float (BatFloat.ceil ((float len) /. (float n))) in
  let rec loop acc = function
    | [] -> L.rev acc
    | lst ->
      let head, tail = L.takedrop m lst in
      loop (head :: acc) tail in
  loop [] l

(* create folds of cross validation; each fold consists in (train, test) *)
let cv_folds n l =
  let test_sets = list_nparts n l in
  assert(n = L.length test_sets);
  let rec loop acc prev curr =
    match curr with
    | [] -> acc
    | x :: xs ->
      let before_after = L.flatten (L.rev_append prev xs) in
      let prev' = x :: prev in
      let train_test = (before_after, x) in
      let acc' = train_test :: acc in
      loop acc' prev' xs in
  loop [] [] test_sets

(* find best (e, C) configuration by R2 maximization *)
let best_r2 l =
  L.fold_left (fun
                ((_best_e, _best_c, best_r2) as best)
                ((_curr_e, _curr_c, curr_r2) as new_best) ->
                if best_r2 >= curr_r2 then
                  best
                else
                  new_best
              ) (0.0, 0.0, 0.0) l

let log_R2 e c r2 =
  (if      r2 < 0.3 then Log.error
   else if r2 < 0.5 then Log.warn
   else                  Log.info) "(e, C, R2) = %g %g %.3f" e c r2

(* return the best parameter configuration (epsilon, C, k) found *)
let optimize_regr verbose ncores es cs train test =
  let ecs = L.cartesian_product es cs in
  let e_c_r2s =
    Parany.Parmap.parmap ncores (fun (e, c) ->
        let act, preds = single_train_test_regr verbose Discard e c train test in
        let r2 = Cpm.RegrStats.r2 act preds in
        log_R2 e c r2;
        (e, c, r2)
      ) ecs in
  best_r2 e_c_r2s

(* like optimize_regr, but using NxCV *)
let optimize_regr_nfolds ncores verbose nfolds es cs train =
  let train_tests = Cpm.Utls.cv_folds nfolds train in
  let ecs = L.cartesian_product es cs in
  let e_c_r2s =
    Parany.Parmap.parmap ncores (fun (e, c) ->
        let all_act_preds =
          L.map (fun (train', test') ->
              single_train_test_regr verbose Discard e c train' test'
            ) train_tests in
        let acts, preds =
          let xs, ys = L.split all_act_preds in
          (L.concat xs, L.concat ys) in
        let r2 = Cpm.RegrStats.r2 acts preds in
        log_R2 e c r2;
        (e, c, r2)
      ) ecs in
  best_r2 e_c_r2s

let mol_of_lines lines =
  L.mapi liblinear_line_to_FpMol lines

let parmap2 ncores f2 l1 l2 =
  Parany.Parmap.parmap ncores (fun (x, y) -> f2 x y) (L.combine l1 l2)

let single_train_test_regr_nfolds verbose nfolds nprocs e c train =
  let train_tests = Cpm.Utls.cv_folds nfolds train in
  let all_act_preds_points =
    Parany.Parmap.parmap nprocs (fun (train', test') ->
        let acts, preds =
          single_train_test_regr verbose Discard e c train' test' in
        ((acts, preds), [])
      ) train_tests in
  let all_act_preds, points = L.split all_act_preds_points in
  let xs, ys = L.split all_act_preds in
  (L.concat xs, L.concat ys, L.concat points)

let dump_AD_points fn points' =
  let points = A.of_list points' in
  A.stable_sort (fun (d1,_,_) (d2,_,_) -> BatFloat.compare d1 d2) points;
  LO.with_out_file fn (fun out ->
      A.iter (fun (d, act, pred) ->
          fprintf out "%f %f %f\n" d act pred
        ) points
    )

(* instance-wise normalization *)
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

let prepend_scores_by_names
    verbose quiet_option do_classification test_fn model_fn output_fn =
  let tmp_csv_fn = pairs_to_csv verbose do_classification test_fn in
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

let prod_predict_regr
    verbose pairs do_classification model_fn test_fn output_fn =
  let quiet_option = if not verbose then "-q" else "" in
  if pairs then
    prepend_scores_by_names
      verbose quiet_option do_classification test_fn model_fn output_fn
  else
    Utls.run_command ~debug:verbose
      (sprintf "%s %s %s %s %s"
         svm_predict quiet_option test_fn model_fn output_fn)

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
     0.01; 0.02; 0.05;
     0.1; 0.2; 0.5;
     1.; 2.; 5.;
     10.; 20.; 50.]
  | Some range_str ->
    L.map float_of_string
      (S.split_on_char ',' range_str)

let decode_k_range (maybe_range_str: string option): int list =
  match maybe_range_str with
  | None ->
    (* default k range *)
    [1; 2; 5; 10; 20; 50]
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
  | (None, None, None) -> failwith "Svmwrap.epsilon_range: no e and no esteps"
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
    LO.map preds_fn float_of_string

let lines_of_file pairs2csv do_classification instance_wise_norm fn =
  let maybe_normalized_lines =
    if instance_wise_norm then
      LO.map fn normalize_line
    else
      LO.lines_of_file fn in
  if pairs2csv then
    L.map (atom_pairs_line_to_csv do_classification) maybe_normalized_lines
  else
    maybe_normalized_lines

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
              [-o <filename>]: predictions output file\n  \
              [-np <int>]: ncores\n  \
              [-c <float>]: fix C\n  \
              [-e <float>]: fix epsilon (for SVR);\n  \
              (0 <= epsilon <= max_i(|y_i|))\n  \
              [--iwn]: turn ON instance-wise-normalization\n  \
              [--no-plot]: no gnuplot\n  \
              [{-n|--NxCV} <int>]: folds of cross validation\n  \
              [-q]: quiet liblinear\n  \
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
              [--regr]: regression (SVR); also, implied by -e and --scan-e\n  \
              [--e-range <float>:<int>:<float>]: specific range for e\n  \
              (semantic=start:nsteps:stop)\n  \
              [--c-range <float,float,...>] explicit scan range for C \n  \
              (example='0.01,0.02,0.03')\n  \
              [--dump-AD <filename>]: dump AD points to file\n  \
              (also requires --regr, --pairs and n>1)\n"
       Sys.argv.(0);
     exit 1);
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
  let ad_points_fn = CLI.get_string_def ["--dump-AD"] args "/dev/null" in
  let compute_AD = ad_points_fn <> "/dev/null" in
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
  let fixed_c = CLI.get_float_opt ["-c"] args in
  let e_range_str = CLI.get_string_opt ["--e-range"] args in
  let c_range_str = CLI.get_string_opt ["--c-range"] args in
  let quiet = CLI.get_set_bool ["-q"] args in
  let instance_wise_norm = CLI.get_set_bool ["--iwn"] args in
  Utls.enforce (not (L.mem "-e" args && L.mem "--scan-e" args))
    "Svmwrap: -e and --scan-e are exclusive";
  let maybe_epsilon = CLI.get_float_opt ["-e"] args in
  let maybe_esteps = CLI.get_int_opt ["--scan-e"] args in
  let do_regression =
    CLI.get_set_bool ["--regr"] args ||
    Opt.is_some maybe_epsilon || Opt.is_some maybe_esteps ||
    Opt.is_some e_range_str in
  let do_classification = not do_regression in
  let no_gnuplot = CLI.get_set_bool ["--no-plot"] args in
  CLI.finalize (); (* ------------------------------------------------------ *)
  let verbose = not quiet in
  (* scan C? *)
  let cs = match fixed_c with
    | Some c -> [c]
    | None ->
      if scan_C || BatOption.is_some c_range_str then
        decode_c_range c_range_str
      else [1.0] in
  (* e-range? *)
  let maybe_es = decode_e_range e_range_str in
  begin match model_cmd with
    | Restore_from models_fn ->
      if do_regression then
        begin
          prod_predict_regr
            verbose pairs do_classification models_fn input_fn output_fn;
          let acts = read_IC50s_from_train_fn pairs input_fn in
          let preds = read_IC50s_from_preds_fn pairs output_fn in
          let r2 = Cpm.RegrStats.r2 acts preds in
          let rmse = Cpm.RegrStats.rmse acts preds in
          let title_str =
            sprintf "T=%s N=%d R2=%.3f RMSE=%.3f" input_fn (L.length preds) r2 rmse in
          (if not no_gnuplot then
             Gnuplot.regr_plot title_str acts preds
          )
        end
      else
        failwith "not do_regression: not implemented yet"
    | Save_into (_)
    | Discard ->
      match maybe_train_fn, maybe_valid_fn, maybe_test_fn with
      | (None, None, None) ->
        begin
          (* randomize lines *)
          let all_lines =
            L.shuffle ~state:rng
              (lines_of_file pairs
                 do_classification instance_wise_norm input_fn) in
            let nb_lines = L.length all_lines in
            (* partition *)
            let train_card =
              BatFloat.round_to_int (train_p *. (float nb_lines)) in
            let train, test = L.takedrop train_card all_lines in
            if do_regression then
              begin
                let best_e, best_c, best_r2 =
                  let epsilons =
                    epsilon_range maybe_epsilon maybe_esteps maybe_es train in
                  if nfolds = 1 then
                    optimize_regr verbose ncores epsilons cs train test
                  else
                    optimize_regr_nfolds
                      ncores verbose nfolds epsilons cs all_lines in
                let actual, preds =
                  if nfolds = 1 then
                    single_train_test_regr
                      verbose model_cmd best_e best_c train test
                  else
                    let actual', preds', ad_points =
                      single_train_test_regr_nfolds
                        verbose nfolds ncores best_e best_c
                        all_lines in
                    (if compute_AD then
                       dump_AD_points ad_points_fn ad_points
                    );
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
                  sprintf "T=%s nfolds=%d e=%g C=%g R2=%.3f RMSE=%.3f"
                    input_fn nfolds best_e best_c best_r2 rmse in
                Log.info "%s" title_str;
                if not no_gnuplot then
                  Gnuplot.regr_plot title_str actual preds
              end
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
