(* Copyright (C) 2021, Francois Berenger

   Tsuda Laboratory, Tokyo University,
   5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan. *)

open Printf

module Fn = Filename
module L = BatList
module LO = Line_oriented
module Log = Dolog.Log
module Stats = Cpm.RegrStats

let protect_underscores title =
  BatString.nreplace ~str:title ~sub:"_" ~by:"\\_"

let regr_plot title actual preds =
  let x_min, x_max = L.min_max ~cmp:BatFloat.compare actual in
  let y_min, y_max = L.min_max ~cmp:BatFloat.compare preds in
  let xy_min = min x_min y_min in
  let xy_max = max x_max y_max in
  let data_fn = Fn.temp_file ~temp_dir:"/tmp" "RFR_regr_data_" ".txt" in
  LO.with_out_file data_fn (fun out ->
      L.iter (fun (x, y) ->
          fprintf out "%f %f\n" x y
        ) (L.combine actual preds)
    );
  let plot_fn = Fn.temp_file ~temp_dir:"/tmp" "RFR_regr_plot_" ".gpl" in
  LO.lines_to_file plot_fn
    ["set xlabel 'actual'";
     "set ylabel 'predicted'";
     "set xtics out nomirror";
     "set ytics out nomirror";
     sprintf "set xrange [%f:%f]" xy_min xy_max;
     sprintf "set yrange [%f:%f]" xy_min xy_max;
     "set key left";
     "set size square";
     sprintf "set title '%s'" (protect_underscores title);
     "g(x) = x";
     "f(x) = a*x + b";
     sprintf "fit f(x) '%s' u 1:2 via a, b" data_fn;
     "plot g(x) t 'perfect' lc rgb 'black', \\";
     sprintf "'%s' using 1:2 not, \\" data_fn;
     "f(x) t 'fit'"];
  (* sprintf "'%s' using 1:2:($2-$3):($2+$3) w errorbars \
    *          t 'n=%d r2=%.2f', \\" data_fn nb_trees r2; *)
  ignore(Sys.command (sprintf "gnuplot --persist %s" plot_fn))
