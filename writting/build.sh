
if [[ $# -ge 1 && $1 = "clean" ]]; then
rm main.aux
rm main.log
rm main.tex
fi

pandoc  main.md -o main.tex --from markdown --template eisvogel.tex --listings
xelatex main.tex
# shellcheck disable=SC2046
DATETIME=$(date -I minutes)
mv main.pdf "$DATETIME.pdf"