#!/usr/bin/env bash

SQLITE="files/variations/base_base_2026-03-05_19-39-04.sqlite"
CSV="files/variations/base_base_2026-03-05_19-39-04_motivation.csv"
JSON="files/base.json"

OUTDIR="frames"
mkdir -p "$OUTDIR"

START=0
END=100
STEP=10

i=0
for frame in $(seq $START $STEP $END); do
    printf "Generating frame %d\n" "$frame"

    # python3 scripts/plot_rank_frame.py \
    #     --sqlite "$SQLITE" \
    #     --motivation-csv "$CSV" \
    #     --json "$JSON" \
    #     --frame "$frame" \
    #     --out "plot_rank_frame"


    # assume script outputs plot_rank_frame.png
#    mv plot_rank_frame_*.png "$OUTDIR/frame_$(printf "%05d" $i).png"

    ((i++))
done

echo "Building video..."

mv plot_rank*.png $OUTDIR
ffmpeg -framerate 30 -i "$OUTDIR/plot_rank_frame_%05d.png" \
    -c:v libx264 -pix_fmt yuv420p \
    rank_animation.mp4

echo "Video written to rank_animation.mp4"
