TIME_BINS="0 100, 500 600, 900 1000"

# Replace spaces with underscores and remove commas
TIME_BINS="${TIME_BINS// /_}"
TIME_BINS="${TIME_BINS//,/}"

echo $TIME_BINS
