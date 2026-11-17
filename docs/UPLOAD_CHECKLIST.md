# Upload checklist (team Montreal)

Deadline: **Tue 2026-11-17, 17:59 EST** (23:59 CET). Target upload: 16:30 EST.

## 1. The file and its hash

The submission file is `~/mtah-data/out/Montreal.csv`, written by
`scripts/release_check.py` (or by `python -m mtah.predict --config configs/final.json`).

<!-- the captain updates this line at each promotion, together with configs/final.json and README -->
```
EXPECTED_SHA256  a97c88ec23a6cd92ea86133e80e56b87dbeeafd4efed9e28804386e3afb918ab   (configs/final.json = v1q)
```

Print the hash of the file you are about to upload and compare it with the line above:

```bash
shasum -a 256 ~/mtah-data/out/Montreal.csv
```

Validate that exact file with the official checker (prints `valid` or the reason):

```bash
cd <repo> && PYTHONPATH=src .venv/bin/python -c "from mtah.submit import validate; print(validate('$HOME/mtah-data/out/Montreal.csv', 199319) or 'valid')"
```

## 2. The two files to upload

| form field | file |
|---|---|
| Possession file | `~/mtah-data/out/Montreal.csv` (~1.9 MB) |
| Short technical document | `docs/Montreal.md` copied as **`Montreal.md`**, or a PDF of it named **`Montreal.pdf`** |

The organisers name board rows after the file, so both files must be named `Montreal.*`.

To get `Montreal.md`, run `cp <repo>/docs/Montreal.md ~/Desktop/Montreal.md`. There is no pandoc
on this machine, so for a PDF, open the Markdown in a viewer and use Print → Save as PDF. The
form accepts `.md` as it is.

## 3. The form, fields in order

The submission URL comes by email to registered participants. You need to be signed in to a
Google account.

1. **Team Name:** `Montreal`, exactly as in the registration form.
2. **Possession file:** `Montreal.csv`.
3. **Public code repository URL:** `https://github.com/ibra049/mtah-2027`
4. **Short technical document:** `Montreal.md` (or `Montreal.pdf`).

After submitting, **keep the "edit your response" link**. Only one response per team counts:
the last one before the deadline, and it can be edited.

## 4. Make the repository public

1. Push `main` (it must contain the commit whose `configs/final.json` produced the hash above).
2. On GitHub, open Settings → General → Danger Zone → Change visibility → Public.
3. Check it while logged out: open `https://github.com/ibra049/mtah-2027` in a private window.
   The README must show, and `configs/final.json` must match the uploaded file.
4. The repository must contain no match data. `.gitignore` excludes it, and the history was
   scanned clean.

## 5. Last checks

- [ ] The `shasum` output matches `EXPECTED_SHA256` above.
- [ ] Official validation prints `valid`.
- [ ] The uploaded files are named `Montreal.csv` and `Montreal.md` / `Montreal.pdf`.
- [ ] The repository is public and readable while logged out, at the commit that made the file.
- [ ] You have the form's edit link.
