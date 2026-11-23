import os, random, shutil
src = 'nastaliq_data'
dst = 'EasyOCR/trainer/all_data'
rows = []
for line in open(f'{src}/labels.tsv', encoding='utf-8'):
    line = line.rstrip('\n')
    if not line.strip():
        continue
    fn, text = line.split('\t', 1)
    if os.path.exists(f'{src}/images/{fn}'):
        rows.append((fn, text.strip()))
random.seed(42)
random.shuffle(rows)
n_val = len(rows) // 10
splits = {'urdu_val': rows[:n_val], 'urdu_train': rows[n_val:]}
chars = set()
for name, items in splits.items():
    os.makedirs(f'{dst}/{name}', exist_ok=True)
    with open(f'{dst}/{name}/labels.csv', 'w', encoding='utf-8', newline='') as f:
        f.write('filename,words\n')
        for fn, text in items:
            shutil.copy(f'{src}/images/{fn}', f'{dst}/{name}/{fn}')
            f.write(f'{fn},{text[::-1]}\n')
            chars.update(text)
open('charset.txt', 'w', encoding='utf-8').write(''.join(sorted(chars)))
print({k: len(v) for k, v in splits.items()}, 'unique chars:', len(chars))
