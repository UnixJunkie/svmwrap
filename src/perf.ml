(* Copyright (C) 2021, Francois Berenger

   Tsuda Laboratory,
   Tokyo University,
   5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan. *)

module A = BatArray
module Fn = Filename
module L = BatList

let compose f g x =
  f (g x)

module Make (SL: Cpm.MakeROC.SCORE_LABEL) = struct

  module ROC = Cpm.MakeROC.Make(SL)

  (* what is the proportion of actives if we keep only molecules
     which score above the given threshold.
     Returns the curve EF = f(score_threshold);
     i.e. a list of (threshold, EF) values.
     Scores must have been normalized. *)
  let actives_portion_plot score_labels =
    (* because thresholds are normalized *)
    let thresholds = L.frange 0.0 `To 1.0 51 in
    let nb_actives = Utls.list_filter_count SL.get_label score_labels in
    let nb_decoys =
      Utls.list_filter_count (compose not SL.get_label) score_labels in
    let empty, rev_res =
      L.fold_left (fun (to_process, acc) t ->
          let to_process' =
            L.filter (fun sl -> SL.get_score sl > t) to_process in
          let card_act =
            Utls.list_filter_count SL.get_label to_process' in
          let card_dec =
            Utls.list_filter_count (compose not SL.get_label) to_process' in
          let n = L.length to_process' in
          let ef =
            if card_act = 0 || n = 0 then
              0.0 (* there are no more actives above this threshold:
                     the EF falls down to 0.0 (threshold too high) *)
            else (* regular EF formula *)
              (float card_act) /. (float n) in
          let rem_acts = (float card_act) /. (float nb_actives) in
          let rem_decs = (float card_dec) /. (float nb_decoys) in
          (to_process', (t, ef, rem_acts, rem_decs) :: acc)
        ) (score_labels, []) thresholds in
    assert(empty = []);
    (nb_actives, nb_decoys, L.rev rev_res)

  let actives_portion_plot_a score_labels =
    (* because thresholds are normalized *)
    let thresholds = L.frange 0.0 `To 1.0 51 in
    let nb_actives = Utls.array_count SL.get_label score_labels in
    let nb_decoys = (A.length score_labels) - nb_actives in
    let rev_res =
      L.fold_left (fun acc t ->
          let n =
            Utls.array_count
              (fun sl -> SL.get_score sl > t)
              score_labels in
          let card_act =
            Utls.array_count
              (fun sl -> (SL.get_label sl) && (SL.get_score sl > t))
              score_labels in
          let card_dec = n - card_act in
          let ef =
            if card_act = 0 || n = 0 then
              0.0 (* there are no more actives above this threshold:
                     the EF falls down to 0.0 (threshold too high) *)
            else (* regular EF formula *)
              (float card_act) /. (float n) in
          let rem_acts = (float card_act) /. (float nb_actives) in
          let rem_decs = (float card_dec) /. (float nb_decoys) in
          (t, ef, rem_acts, rem_decs) :: acc
        ) [] thresholds in
    (nb_actives, nb_decoys, L.rev rev_res)

end
