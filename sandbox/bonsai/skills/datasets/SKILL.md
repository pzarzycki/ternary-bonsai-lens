---
name: datasets
description: Use when downloading or processing large data (dumps, corpora, CSV/JSON/XML exports, compressed files): inspect first, stream, subset, staged and resumable processing.
---

# Working with large datasets

1. **Download once** to `data/` with `curl -fL --retry 3 -o data/<name> <url>` as its own command. Check the
   size (`ls -lh`) and integrity (`bzip2 -t`, `gzip -t`, `unzip -t`) before using it. Don't download again
   if the file exists.
2. **Inspect before parsing.** Look at a real sample: `bzcat f.bz2 | head -100`, `zcat f.gz | head`,
   `head -c 3000 f`, `jq '.[0]' f.json`, `file f`. Note the record boundaries, fields, encoding, special
   cases (redirects, empty records, markup) and the file's real size. Write the parser for what you saw.
3. **Stream, don't load.** Read compressed files directly: `bz2.open(path, "rt")` or `gzip.open`. For XML use
   `xml.etree.ElementTree.iterparse(f, events=("end",))` and call `elem.clear()` after each record, so memory
   stays flat. Strip namespaces with `tag.rsplit("}", 1)[-1]`. For JSON lines, read line by line.
4. **Subset first.** Develop and test on the first ~1000 records (add a `--limit` option), check the output
   by eye, then run on everything.
5. **Staged and resumable.** Each stage reads one artifact and writes the next (raw → clean text →
   tokens), skipping the work if its output already exists and is newer. Write to a temp file, then
   rename it. Print progress (records, MB/s) with `flush=True`.
6. **Validate outputs.** Check record counts, sizes and a few random samples, and assert the things you rely
   on (e.g. no markup left: `grep -c '{{' out.txt` should be 0).

Wiki dumps: text sits in `<page><revision><text>`. Skip `<redirect>` pages, `{{…}}` templates, `{|…|}` tables
and `[[File:…]]` and `<ref>` tags. Keep link text from `[[target|text]]`.
