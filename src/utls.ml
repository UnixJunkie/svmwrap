(* Copyright (C) 2021, Francois Berenger

   Tsuda Laboratory,
   Tokyo University,
   5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan. *)

(* not using Batteries !!! Dont! I want fast IOs.
   Or we have to prefix many things with Legacy.XXX *)
open Printf

module A = BatArray
module Fn = Filename
module L = BatList
module LO = Line_oriented
module Log = Dolog.Log

type filename = string

let tap f x =
  f x;
  x

let fst3 (a, _, _) = a

let create_tmp_filename () =
  let res = Fn.temp_file
      ~temp_dir:"/tmp" "" (* no_prefix *) "" (* no_suffix *) in
  (* tap (Log.info "create_tmp_filename: %s") res; *)
  res

let mkfifo (fn: filename): unit =
  Unix.mkfifo fn 0o600

(* abort if condition is not met *)
let enforce (condition: bool) (err_msg: string): unit =
  if not condition then
    failwith err_msg

let array_to_file (fn: filename) (f: 'a -> string) (a: 'a array): unit =
  LO.with_out_file fn (fun out ->
      A.iter (fun x ->
          fprintf out "%s\n" (f x)
        ) a
    )

(* get one bootstrap sample of size 'nb_samples' using
   sampling with replacement *)
let array_bootstrap_sample rng nb_samples a =
  let n = Array.length a in
  assert(nb_samples <= n);
  A.init nb_samples (fun _ ->
      A.unsafe_get a (Random.State.int rng n)
    )

(* skip 'nb' blocks from file being read *)
let skip_blocks nb read_one input =
  if nb = 0 then ()
  else
    let () = assert(nb > 0) in
    for _ = 1 to nb do
      ignore(read_one input)
    done

(* get the first line (stripped) output by given command *)
let get_command_output (cmd: string): string =
  Log.info "get_command_output: %s" cmd;
  let _stat, output = BatUnix.run_and_read cmd in
  match BatString.split_on_char '\n' output with
  | first_line :: _others -> first_line
  | [] -> (Log.fatal "get_command_output: no output for: %s" cmd; exit 1)

(* run the given command in a sub process (in parallel to the current process)
   and returns its pid so that we can wait for it later *)
let fork_out_cmd (cmd: string): int =
  Log.info "fork_out_cmd: %s" cmd;
  match Unix.fork () with
  | 0 -> (* child process *) exit (Sys.command cmd)
  | -1 -> (* error *) (Log.fatal "fork_out_cmd: fork failed"; exit 1)
  | pid -> pid

(* return full path of command, if found in PATH, none else *)
let command_exists (cmd: string): string option =
  let where_is_cmd = "which " ^ cmd in
  if Unix.system (where_is_cmd ^ " 2>&1 > /dev/null") = Unix.WEXITED 0 then
    Some (get_command_output where_is_cmd)
  else
    None

let run_command ?(debug = false) (cmd: string): unit =
  if debug then Log.info "run_command: %s" cmd;
  match Unix.system cmd with
  | Unix.WSIGNALED _ -> (Log.fatal "run_command: signaled: %s" cmd; exit 1)
  | Unix.WSTOPPED _ -> (Log.fatal "run_command: stopped: %s" cmd; exit 1)
  | Unix.WEXITED i when i <> 0 ->
    (Log.fatal "run_command: exit %d: %s" i cmd; exit 1)
  | Unix.WEXITED _ (* i = 0 then *) -> ()

let get_env (env_var: string): string option =
  try Some (Sys.getenv env_var)
  with Not_found -> None

(* look for exe in PATH then given env. var *)
let find_command (exe: string) (env_var: string): string option =
  match command_exists exe with
  | Some cmd -> Some cmd
  | None ->
    match get_env env_var with
    | Some cmd -> Some cmd
    | None -> (Log.warn "%s not found in PATH; \
                         put it in your PATH or setup the \
                         %s env. var. to point to it" exe env_var;
               None)

let filename_is_absolute fn =
  not (Fn.is_relative fn)

let relative_to_absolute fn =
  if Fn.is_relative fn then
    let cwd = Sys.getcwd () in
    Fn.concat cwd fn
  else
    fn

(* remove the prefix if it is there, or do nothing if it is not *)
let remove_string_prefix prfx str =
  if BatString.starts_with str prfx then
    let prfx_len = String.length prfx in
    BatString.tail str prfx_len
  else
    str

let string_contains super sub =
  try let _i = BatString.find super sub in true
  with Not_found -> false

let os_is_Mac_OS () =
  string_contains (get_command_output "uname -a") "Darwin"

let may_apply f = function
  | Some x -> f x
  | None -> ()

(* returns true if we could create the file; false else (already there) *)
let lock_file_for_writing (fn: filename): bool =
  try
    let fd = Unix.(openfile fn [O_CREAT; O_EXCL; O_WRONLY] 0o600) in
    Unix.close fd;
    true
  with Unix.Unix_error _ -> false

exception Enough_times

(* accumulate the result of calling 'f' 'n' times *)
let n_times n f =
  let i = ref 0 in
  BatList.unfold_exc (fun () ->
      if !i = n then
        raise Enough_times
      else
        let res = f () in
        incr i;
        res
    )

(* measure time spent in f (seconds) *)
let wall_clock_time f =
  let start = Unix.gettimeofday () in
  let res = f () in
  let stop = Unix.gettimeofday () in
  let delta_t = stop -. start in
  (delta_t, res)

(* the identity function *)
let id x = x

(* enforce filename uses one of the allowed extensions *)
let enforce_file_extension allowed_exts fn =
  assert(L.exists (BatString.ends_with fn) allowed_exts)

(* Pi math constant *)
let m_pi = 4.0 *. atan 1.0

let prepend x xs =
  xs := x :: !xs

(* test (lb <= x <= hb) *)
let in_bounds lb x hb =
  x >= lb && x <= hb

let list_medianf (l: float list): float =
  let xs = Array.of_list l in
  Array.sort BatFloat.compare xs;
  let n = Array.length xs in
  if n mod 2 = 1 then
    xs.(n/2)
  else
    0.5 *. (xs.(n/2) +. xs.(n/2 - 1))
(*$T list_medianf
   list_medianf [1.;2.;3.;4.;5.] = 3.0
   list_medianf [1.;2.;3.;4.] = 2.5
*)

let string_of_array ?pre:(pre = "[|") ?sep:(sep = ";") ?suf:(suf = "|]")
    to_str a =
  let buff = Buffer.create 80 in
  Buffer.add_string buff pre;
  Array.iteri (fun i x ->
      if i > 0 then Buffer.add_string buff sep;
      Buffer.add_string buff (to_str x)
    ) a;
  Buffer.add_string buff suf;
  Buffer.contents buff

let marshal_to_string x =
  Marshal.(to_string x [No_sharing])

let unmarshal_from_string s =
  Marshal.from_string s 0

let is_odd i =
  i mod 2 = 1

let is_even i =
  i mod 2 = 0

(* like the cut unix command *)
let cut d f line =
  let splitted = BatString.split_on_char d line in
  BatList.at splitted f

let get_ncores () =
  int_of_string (get_command_output "getconf _NPROCESSORS_ONLN")

let int_of_bool = function
  | true -> 1
  | false -> 0

let bool_of_int = function
  | 1 -> true
  | 0 -> false
  | _ -> assert(false)

(* test that 'y' is within 'x' +/- 'epsilon';
   i.e. y \in [x - epsilon, x + epsilon] *)
let approx_equal epsilon x y =
  (y >= x -. epsilon) &&
  (y <= x +. epsilon)

(* proper NaN/nan testing *)
let is_nan x =
  match classify_float x with
  | FP_nan -> true
  | _ -> false

(* some statistics *)
let favg = function
  | [] -> 0.0 (* protect against empty list *)
  | xs -> L.favg xs

(* population standard deviation *)
let stddev (l: float list): float =
  let n, sx, sx2 =
    List.fold_left (fun (n, sx, sx2) x ->
        (n +. 1., sx +. x, sx2 +. (x *.x))
      ) (0., 0., 0.) l
  in
  sqrt ((sx2 -. (sx *. sx) /. n) /. n)
(* stddev [2.; 4.; 4.; 4.; 5.; 5.; 7.; 9.] = 2.0 *)

let fincr_by (xref: float ref) (dx: float): unit =
  xref := !xref +. dx

let min_max x y =
  if x <= y then
    (x, y)
  else
    (y, x)

let array_count p a =
  let i = ref 0 in
  A.iter (fun x -> if p x then incr i) a;
  !i

let list_combine3 l1 l2 l3 =
  let l1_l2 = L.combine l1 l2 in
  L.map2 (fun (x, y) z -> (x, y, z)) l1_l2 l3
