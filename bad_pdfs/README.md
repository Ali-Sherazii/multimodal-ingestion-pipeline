# Bad PDFs test set

These are 22 deliberately difficult PDFs for stress-testing the ingestion pipeline. Each file targets one edge case. Ingest the whole folder in one go:

```
ingest run bad_pdfs/
```

Then compare the run report against the **Expected** column below. Some files should fail or be skipped. That's correct behaviour, as long as the batch keeps going and the reason is clear.

Note: `.md` files are supported, so this README will be ingested too. Move it out of the folder first if you don't want it in the database.

| # | File | What's nasty about it | Expected | What to check |
|---|---|---|---|---|
| 01 | two_column_layout | Two text columns per page | processed | Each "LEFT-OR-RIGHT MARKER n" is followed by its own section's text, not text from the other column |
| 02 | scanned_clean | Image only, no text layer | processed (via OCR) | `ingest search "sick leave"` finds it, with page 1 |
| 03 | scanned_terrible_quality | Rotated, blurry, speckled, low-res, grey background | processed, with OCR errors | How much of the text survives? Compare with 02 |
| 04 | mixed_digital_and_scanned | Page 1 digital, page 2 scanned | processed | Chunks from both pages exist, with page numbers 1 and 2 |
| 05 | complex_tables | Merged header cells, a borderless table, a 120-row table across 3 pages | processed | Tables come out as Markdown; check whether the long table's header row survives in later chunks |
| 06 | repeating_headers_footers | "CONFIDENTIAL…" header and "Page n of 8" footer on every page | processed | Neither line should appear in any chunk (`ingest show --file 06`) |
| 07 | rotated_and_landscape_pages | Pages rotated 90° and 180°, plus a landscape page | processed | Rotated pages' text is readable, not garbled or missing |
| 08 | password_protected | AES-256 encrypted (password `rapids123`) | **failed** | The error message clearly says it's encrypted, and the batch continues |
| 09 | truncated_corrupt | File cut off at one-third of its size | **failed** (or partial) | No crash; clear reason in the report |
| 10 | random_bytes_damaged | 400 random bytes overwritten | **failed** or partial | Same as 09; if partial, is the extracted text garbage? |
| 11 | fake_actually_text | A text file renamed to `.pdf` | **skipped** | Reason: content does not match its .pdf extension |
| 12 | empty_zero_bytes | 0 bytes | **skipped** | Reason: empty file |
| 13 | blank_pages_no_text | Valid PDF, 3 pages with only a border | **failed** | Reason: no text could be extracted |
| 14 | charts_logos_photos_as_images | Bar chart, tiny logo and a noise "photo", all as images | processed | Chart numbers/labels via OCR? Figure captions present? The small logo is probably missed; the photo yields nothing |
| 15 | garbage_hidden_ocr_layer | Clean scan, but with an invisible text layer full of gibberish (like a bad old OCR) | processed, **likely with garbage chunks** | Does Docling trust the bad layer? Retry with `pdf_force_full_page_ocr: true` and compare |
| 16 | hidden_white_and_tiny_text | White-on-white "ignore previous instructions…" text and a 1pt fake price | processed, **hidden text ingested** | Shows that invisible text reaches the database; a risk once a chatbot reads these chunks |
| 17 | urdu_english_mixed_rtl | Urdu (right-to-left) mixed with English, stored in visual order | processed | Is the Urdu extracted in the right order, or reversed? Search quality will be poor with the English-only embedding model |
| 18 | ligatures_hyphenation_odd_chars | ﬁ/ﬂ/ﬃ ligatures, words hyphenated across lines, soft hyphens, zero-width and non-breaking spaces, smart quotes | processed | Ligatures become plain letters (NFKC); invisible characters are removed; hyphenated words like "per- formance" probably stay split |
| 19 | large_400_pages | 400 pages of text | processed (slow) | Timings per stage in the report; memory use |
| 20 | duplicate_of_01_renamed | Byte-identical copy of 01 | **skipped** | Reason: already ingested, unchanged. Note that search results will only ever show 01's path |
| 21 | `Final Report (v2) – ڈرافٹ #1 & copy` | Spaces, brackets, en dash, Urdu, `#` and `&` in the file name | processed | Path handled correctly everywhere: report, `show`, `delete` |
| 22 | dense_tiny_font_table | 55-row, 9-column table in 5.5pt font, landscape | processed | Is the table structure correct at that size? How is it split into chunks? |

## Regenerating

`make_bad_pdfs.py` rebuilds the whole set:

```
pip install reportlab pillow pypdf arabic-reshaper python-bidi
python make_bad_pdfs.py bad_pdfs
```

It uses DejaVu and FreeSerif fonts from `/usr/share/fonts` (installed by default on most Linux systems). On Windows, change `FONT_DIR` and the FreeSerif path to fonts that exist on your machine, with an Urdu-capable font for file 17.
