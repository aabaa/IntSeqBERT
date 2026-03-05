#!/usr/bin/env bash
# make_arxiv.sh — Create arXiv submission tarball for IntSeqBERT paper.
# Usage: bash scripts/make_arxiv.sh
# Output: arxiv_submission.tar.gz in paper/2026-03/

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_NAME="arxiv_submission"
TMP_DIR="$(mktemp -d)"
DEST="${TMP_DIR}/${OUT_NAME}"

echo "Paper directory: ${PAPER_DIR}"
echo "Staging in:      ${DEST}"

mkdir -p "${DEST}/sections" "${DEST}/figures"

# --- LaTeX source files ---
cp "${PAPER_DIR}/main.tex"      "${DEST}/"
cp "${PAPER_DIR}/references.bib" "${DEST}/"
cp "${PAPER_DIR}/sections/"*.tex "${DEST}/sections/"

# --- Figures: only those referenced in the paper ---
# PDF preferred (vector); fall back to PNG when PDF absent.
USED_FIGS=(
    fig1_architecture
    fig2_nig_spectrum
    fig2b_nig_vs_phi
    fig3_scaling
    fig4_learning_curves
    fig5_magnitude_scatter
)

for fig in "${USED_FIGS[@]}"; do
    if [ -f "${PAPER_DIR}/figures/${fig}.pdf" ]; then
        cp "${PAPER_DIR}/figures/${fig}.pdf" "${DEST}/figures/"
        echo "  included: figures/${fig}.pdf"
    elif [ -f "${PAPER_DIR}/figures/${fig}.png" ]; then
        cp "${PAPER_DIR}/figures/${fig}.png" "${DEST}/figures/"
        echo "  included: figures/${fig}.png (no PDF available)"
    else
        echo "  WARNING: figures/${fig} not found!" >&2
    fi
done

# --- Create tarball ---
OUTPUT_TAR="${PAPER_DIR}/${OUT_NAME}.tar.gz"
tar -czf "${OUTPUT_TAR}" -C "${TMP_DIR}" "${OUT_NAME}"
rm -rf "${TMP_DIR}"

echo ""
echo "Created: ${OUTPUT_TAR}"
echo "Contents:"
tar -tzf "${OUTPUT_TAR}"
