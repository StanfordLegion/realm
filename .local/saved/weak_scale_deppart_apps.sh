#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"

build_dir="${BUILD_DIR:-${repo_dir}/build}"
benchmark="${BENCHMARK:-${build_dir}/tests/benchmark}"
out_dir="${OUT_DIR:-${build_dir}}"
log_dir="${LOG_DIR:-${out_dir}/weak_scale_logs}"

reps="${REPS:-3}"
scales="${SCALES:-1 2 3 4}"

ll_fsize="${LL_FSIZE:-15500}"
ll_csize="${LL_CSIZE:-32768}"
ll_zsize="${LL_ZSIZE:-4096}"

# Defaults are reduced weak-scale per-p sizes. The larger paper-scale sizes were
# useful for GPU capacity testing, but they make the CPU path memory-bandwidth
# dominated enough to obscure weak-scaling behavior.
circuit_edges_per_piece="${CIRCUIT_EDGES_PER_PIECE:-4500000}"
pennant_nzx_per_p="${PENNANT_NZX_PER_P:-675}"
pennant_nzy="${PENNANT_NZY:-2096}"
miniaero_gy="${MINIAERO_GY:-1202}"
miniaero_gz="${MINIAERO_GZ:-1202}"

use_srun="${USE_SRUN:-1}"
read -r -a srun_args <<< "${SRUN_ARGS:-srun -n 1 -N 1 -c 40 -p gpu --exclusive}"

mkdir -p "${out_dir}" "${log_dir}"

circuit_csv="${out_dir}/circuit.csv"
pennant_csv="${out_dir}/pennant.csv"
miniaero_csv="${out_dir}/miniaero.csv"

printf 'p,rep,ll_gpu,ll_util,dp_workers,num_nodes,num_edges,num_pieces,field_instance_size,gpu_us,cpu_us,gpu_noisect_us,cpu_isect_us,gpu_isect_us,cpu_noisect_us,speedup\n' > "${circuit_csv}"
printf 'p,rep,ll_gpu,ll_util,dp_workers,nzx,nzy,numpcx,numpcy,num_zones,num_sides,num_points,field_instance_size,gpu_us,cpu_us,gpu_noisect_us,cpu_isect_us,gpu_isect_us,cpu_noisect_us,speedup\n' > "${pennant_csv}"
printf 'p,rep,ll_gpu,ll_util,dp_workers,gx,gy,gz,bx,by,bz,num_cells,num_faces,field_instance_size,gpu_us,cpu_us,gpu_noisect_us,cpu_isect_us,gpu_isect_us,cpu_noisect_us,speedup\n' > "${miniaero_csv}"

extract_result_value()
{
  local key="$1"
  local log="$2"
  awk -F, -v key="${key}" '
    /^RESULT,/ {
      for(i = 1; i <= NF; i++) {
        split($i, a, "=")
        if(a[1] == key) {
          print a[2]
          exit
        }
      }
    }
  ' "${log}"
}

speedup()
{
  local cpu_us="$1"
  local gpu_us="$2"
  awk -v cpu="${cpu_us}" -v gpu="${gpu_us}" 'BEGIN { printf "%.6f", cpu / gpu }'
}

run_benchmark()
{
  local app="$1"
  local mode="$2"
  local p="$3"
  local rep="$4"
  shift 4
  local log="${log_dir}/${app}_${mode}_p${p}_rep${rep}.log"
  local cmd=("${benchmark}" "$app" "$@" \
             -dp:workers "${p}")

  if [[ "${mode}" == "noisect" ]]; then
    cmd+=(-dp:noisectopt)
  elif [[ "${mode}" != "isect" ]]; then
    printf 'unknown mode=%s for app=%s p=%s rep=%s\n' \
      "${mode}" "${app}" "${p}" "${rep}" >&2
    exit 1
  fi

  cmd+=(-ll:gpu "${p}" -ll:util "${p}" -ll:fsize "${ll_fsize}" \
        -ll:csize "${ll_csize}" -ll:zsize "${ll_zsize}")

  printf 'running app=%s mode=%s p=%s rep=%s\n' "${app}" "${mode}" "${p}" "${rep}" >&2
  printf 'command:' > "${log}"
  printf ' %q' "${cmd[@]}" >> "${log}"
  printf '\n' >> "${log}"

  local status=0
  if [[ "${use_srun}" == "1" ]]; then
    "${srun_args[@]}" "${cmd[@]}" >> "${log}" 2>&1 || status=$?
  else
    "${cmd[@]}" >> "${log}" 2>&1 || status=$?
  fi

  if [[ "${status}" != "0" ]]; then
    printf 'run failed with status=%s for app=%s mode=%s p=%s rep=%s; see %s\n' \
      "${status}" "${app}" "${mode}" "${p}" "${rep}" "${log}" >&2
    tail -n 40 "${log}" >&2 || true
    exit "${status}"
  fi

  if ! grep -q '^RESULT,' "${log}"; then
    printf 'missing RESULT line for app=%s mode=%s p=%s rep=%s; see %s\n' \
      "${app}" "${mode}" "${p}" "${rep}" "${log}" >&2
    tail -n 40 "${log}" >&2 || true
    exit 1
  fi

  printf '%s\n' "${log}"
}

run_circuit()
{
  local p="$1"
  local rep="$2"
  local num_nodes="${p}"
  local num_edges=$((circuit_edges_per_piece * p))
  local num_pieces="${p}"
  local field_instance_size="${circuit_edges_per_piece}"

  local isect_log noisect_log
  isect_log="$(run_benchmark circuit isect "${p}" "${rep}" \
                -n "${num_nodes}" -e "${num_edges}" -p "${num_pieces}" -buffer 100)"
  noisect_log="$(run_benchmark circuit noisect "${p}" "${rep}" \
                  -n "${num_nodes}" -e "${num_edges}" -p "${num_pieces}" -buffer 100)"

  local gpu_us cpu_us gpu_isect_us cpu_noisect_us
  gpu_us="$(extract_result_value gpu_us "${noisect_log}")"
  cpu_us="$(extract_result_value cpu_us "${isect_log}")"
  gpu_isect_us="$(extract_result_value gpu_us "${isect_log}")"
  cpu_noisect_us="$(extract_result_value cpu_us "${noisect_log}")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${p}" "${rep}" "${p}" "${p}" "${p}" "${num_nodes}" "${num_edges}" "${num_pieces}" \
    "${field_instance_size}" "${gpu_us}" "${cpu_us}" "${gpu_us}" "${cpu_us}" \
    "${gpu_isect_us}" "${cpu_noisect_us}" "$(speedup "${cpu_us}" "${gpu_us}")" \
    >> "${circuit_csv}"
}

run_pennant()
{
  local p="$1"
  local rep="$2"
  local nzx=$((pennant_nzx_per_p * p))
  local nzy="${pennant_nzy}"
  local numpcx="${p}"
  local numpcy=1
  local field_instance_size=$((4 * pennant_nzx_per_p * pennant_nzy))

  local isect_log noisect_log
  isect_log="$(run_benchmark pennant isect "${p}" "${rep}" \
                -nzx "${nzx}" -nzy "${nzy}" -p "${p}" \
                -buffer 100)"
  noisect_log="$(run_benchmark pennant noisect "${p}" "${rep}" \
                  -nzx "${nzx}" -nzy "${nzy}" -p "${p}" \
                  -buffer 100)"

  local gpu_us cpu_us gpu_isect_us cpu_noisect_us num_zones num_sides num_points
  gpu_us="$(extract_result_value gpu_us "${noisect_log}")"
  cpu_us="$(extract_result_value cpu_us "${isect_log}")"
  gpu_isect_us="$(extract_result_value gpu_us "${isect_log}")"
  cpu_noisect_us="$(extract_result_value cpu_us "${noisect_log}")"
  num_zones="$(extract_result_value num_zones "${isect_log}")"
  num_sides="$(extract_result_value num_sides "${isect_log}")"
  num_points="$(extract_result_value num_points "${isect_log}")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${p}" "${rep}" "${p}" "${p}" "${p}" "${nzx}" "${nzy}" "${numpcx}" "${numpcy}" \
    "${num_zones}" "${num_sides}" "${num_points}" "${field_instance_size}" \
    "${gpu_us}" "${cpu_us}" "${gpu_us}" "${cpu_us}" \
    "${gpu_isect_us}" "${cpu_noisect_us}" "$(speedup "${cpu_us}" "${gpu_us}")" \
    >> "${pennant_csv}"
}

run_miniaero()
{
  local p="$1"
  local rep="$2"
  local gx="${p}"
  local gy="${miniaero_gy}"
  local gz="${miniaero_gz}"
  local bx="${p}"
  local by=1
  local bz=1
  local field_instance_size=$((((1 + 1) * gy * gz) + ((1) * (gy + 1) * gz) + ((1) * gy * (gz + 1))))

  local isect_log noisect_log
  isect_log="$(run_benchmark miniaero isect "${p}" "${rep}" \
                -gx "${gx}" -gy "${gy}" -gz "${gz}" -p "${p}" \
                -buffer 100)"
  noisect_log="$(run_benchmark miniaero noisect "${p}" "${rep}" \
                  -gx "${gx}" -gy "${gy}" -gz "${gz}" -p "${p}" \
                  -buffer 100)"

  local gpu_us cpu_us gpu_isect_us cpu_noisect_us num_cells num_faces
  gpu_us="$(extract_result_value gpu_us "${noisect_log}")"
  cpu_us="$(extract_result_value cpu_us "${isect_log}")"
  gpu_isect_us="$(extract_result_value gpu_us "${isect_log}")"
  cpu_noisect_us="$(extract_result_value cpu_us "${noisect_log}")"
  num_cells="$(extract_result_value num_cells "${isect_log}")"
  num_faces="$(extract_result_value num_faces "${isect_log}")"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${p}" "${rep}" "${p}" "${p}" "${p}" "${gx}" "${gy}" "${gz}" "${bx}" "${by}" "${bz}" \
    "${num_cells}" "${num_faces}" "${field_instance_size}" \
    "${gpu_us}" "${cpu_us}" "${gpu_us}" "${cpu_us}" \
    "${gpu_isect_us}" "${cpu_noisect_us}" "$(speedup "${cpu_us}" "${gpu_us}")" \
    >> "${miniaero_csv}"
}

for rep in $(seq 1 "${reps}"); do
  for p in ${scales}; do
    run_circuit "${p}" "${rep}"
  done
done

for rep in $(seq 1 "${reps}"); do
  for p in ${scales}; do
    run_pennant "${p}" "${rep}"
  done
done

for rep in $(seq 1 "${reps}"); do
  for p in ${scales}; do
    run_miniaero "${p}" "${rep}"
  done
done

printf 'wrote %s\n' "${circuit_csv}"
printf 'wrote %s\n' "${pennant_csv}"
printf 'wrote %s\n' "${miniaero_csv}"
