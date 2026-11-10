#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NB=${NB:-1000000}
NQ=${NQ:-10000}
NL=${NL:-1000000}

human() {
  local n=$1
  if (( n % 1000000 == 0 )); then echo "$((n / 1000000))M"
  elif (( n % 1000 == 0 )); then echo "$((n / 1000))K"
  else echo "$n"; fi
}

le32() {
  local v=$1
  printf "\\x$(printf %02x $((v & 255)))\\x$(printf %02x $(((v >> 8) & 255)))\\x$(printf %02x $(((v >> 16) & 255)))\\x$(printf %02x $(((v >> 24) & 255)))"
}

fetch_fbin() {
  local url=$1 out=$2 n=$3 dim=$4
  local bytes=$((8 + n * dim * 4))
  if [[ -f "$out" && $(stat -c %s "$out") -eq $bytes ]]; then
    echo "  [skip] $(basename "$out") already complete"
    return
  fi
  echo "  [get ] $(basename "$out")  (${n} x ${dim}, $((bytes / 1000000)) MB)  <- ${url}"
  local have=0
  [[ -f "${out}.part" ]] && have=$(stat -c %s "${out}.part")
  if (( have > bytes )); then rm -f "${out}.part"; have=0; fi
  if (( have > 0 )); then
    curl -L --fail --retry 5 --retry-delay 5 -r "${have}-$((bytes - 1))" "$url" >> "${out}.part"
  else
    curl -L --fail --retry 5 --retry-delay 5 -r "0-$((bytes - 1))" -o "${out}.part" "$url"
  fi
  local got
  got=$(stat -c %s "${out}.part")
  if [[ $got -ne $bytes ]]; then
    echo "  [fail] expected ${bytes} bytes, got ${got}" >&2
    exit 1
  fi
  read -r n0 d0 < <(od -An -t d4 -N 8 "${out}.part")
  if [[ $d0 -ne $dim || $n0 -lt $n ]]; then
    echo "  [fail] header says ${n0} x ${d0}, expected at least ${n} x ${dim}" >&2
    exit 1
  fi
  { le32 "$n"; le32 "$dim"; } | dd of="${out}.part" bs=8 count=1 iflag=fullblock conv=notrunc status=none
  mv "${out}.part" "$out"
}

write_meta() {
  cat > "$1/dataset.json" <<EOF
{
  "name": "$2",
  "metric": "$3",
  "dim": $4,
  "nb": ${NB},
  "nq": ${NQ},
  "nl": ${NL},
  "files": {"base": "base.fbin", "query": "query.fbin", "learn": "learn.fbin"},
  "source": "$5"
}
EOF
}

get_t2i() {
  local name="t2i-$(human "$NB")" dir
  dir="${DATA_DIR}/${name}"; mkdir -p "$dir"
  local base_url="https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I"
  echo "== ${name}: Yandex Text-to-Image, 200-d, inner product"
  fetch_fbin "${base_url}/base.1B.fbin"            "$dir/base.fbin"  "$NB" 200
  fetch_fbin "${base_url}/query.public.100K.fbin"  "$dir/query.fbin" "$NQ" 200
  fetch_fbin "${base_url}/query.learn.50M.fbin"    "$dir/learn.fbin" "$NL" 200
  write_meta "$dir" "$name" ip 200 "${base_url}"
}

get_webvid() {
  local name="webvid-$(human "$NB")" dir
  dir="${DATA_DIR}/${name}"; mkdir -p "$dir"
  local base_url="https://zenodo.org/records/11090378/files"
  echo "== ${name}: CLIP WebVid, 512-d, inner product"
  fetch_fbin "${base_url}/clip.webvid.base.2.5M.fbin"    "$dir/base.fbin"  "$NB" 512
  fetch_fbin "${base_url}/webvid.query.10k.fbin"         "$dir/query.fbin" "$NQ" 512
  fetch_fbin "${base_url}/webvid.query.train.2.5M.fbin"  "$dir/learn.fbin" "$NL" 512
  write_meta "$dir" "$name" ip 512 "${base_url}"
}

case "${1:-all}" in
  t2i)    get_t2i ;;
  webvid) get_webvid ;;
  all)    get_t2i; get_webvid ;;
  *) echo "usage: $0 [t2i|webvid|all]" >&2; exit 1 ;;
esac
echo "done."
