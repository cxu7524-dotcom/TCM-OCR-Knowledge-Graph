"""Assemble the dual-track small-character page dataset.

Copies the reviewed pages into 古籍评测集/双轨小字页/<book>/ and writes a manifest
with the concrete example that justified each inclusion. Review outcomes for the
pages that were excluded are recorded too, so the screen is auditable.
"""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / '古籍评测集/医家类'
DST = ROOT / '古籍评测集/双轨小字页'

# status: 收录 = confirmed dual track; 待核 = probable but boundary case; 排除 = reviewed, no dual track
ROWS = [
    # book, page, status, confidence, main, right, left, note
    ('思濟堂方書', 33, '收录', '确认', '川大黃', '二錢半', '酒洗', '第3栏；另有白朮土炒等单轨'),
    ('思濟堂方書', 34, '收录', '确认', '大膠棗', '十枚煮', '去皮核', '既有标注 P34-10；犀角 一錢二分/鎊 P34-11'),
    ('思濟堂方書', 35, '收录', '确认', '白朮', '五錢', '泔炒', '开发样例 dev-35-001，同一框'),
    ('思濟堂方書', 36, '收录', '确认', '正嫩烏藥', '二錢', '打碎', '烏藥順氣湯方'),
    ('思濟堂方書', 37, '收录', '确认', '揀芍', '二錢半', '酒炒', '既有标注 P37-01'),
    ('思濟堂方書', 39, '收录', '确认', '白朮', '四錢', '黃土拌炒', '既有标注 P39-04；川附子 各二錢/製 P39-02'),
    ('思濟堂方書', 42, '收录', '确认', '麥冬', '各三錢', '去心', '生脈散一類方'),
    ('思濟堂方書', 43, '收录', '确认', '巴豆', '五錢去皮', '心研如脂', '既有标注 P43-05'),
    ('思濟堂方書', 45, '收录', '确认', '之', '方見', '中風', '既有标注 P45-19；二錢 滾水泡五次用 P45-17'),
    ('思濟堂方書', 46, '收录', '确认', '半夏', '薑', '製', '既有标注 P46-06'),
    ('思濟堂方書', 49, '收录', '确认', '杏仁', '七枚去皮', '尖研', '既有标注 P49-12'),
    ('思濟堂方書', 50, '收录', '确认', '黃芩', '各一錢半', '酒炒', '既有标注 P50-09'),
    ('温疫方論', 12, '收录', '确认', '澤瀉', '一錢', '五分', '四苓湯；豬苓 一錢/五分 同式'),
    ('思濟堂方書', 38, '待核', '待核', '桃仁', '二十枚', '去皮尖', '欄條放大可見兩軌，末筆與下一字邊界待核'),
    ('思濟堂方書', 51, '待核', '待核', '枳殼', '各一錢半', '麩炒', '人參敗毒散；小字淡，兩軌分界待核'),
    ('思濟堂方書', 30, '排除', '—', '', '', '', '卷首序，無方劑'),
    ('思濟堂方書', 31, '排除', '—', '', '', '', '論說正文，藥名後僅單軌'),
    ('思濟堂方書', 32, '排除', '—', '', '', '', '通關散等：藥名後僅單條小字（劑量或炮制擇一）'),
    ('思濟堂方書', 40, '排除', '—', '', '', '', '澤瀉一錢二分、牡丹皮長注均為單軌'),
    ('思濟堂方書', 41, '排除', '—', '', '', '', '理中湯：人參一錢半、白朮土炒、乾薑各二錢 皆單軌'),
    ('思濟堂方書', 44, '排除', '—', '', '', '', '中風論說'),
    ('思濟堂方書', 47, '排除', '—', '', '', '', '傷寒論說，小字在天頭'),
    ('思濟堂方書', 48, '排除', '—', '', '', '', '論說附加減法'),
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    DST.mkdir(parents=True, exist_ok=True)
    manifest = []
    for book, page, status, conf, main_, right, left, note in ROWS:
        src = SRC / book / f'page_{page}.png'
        assert src.exists(), src
        row = {'book': book, 'page': f'page_{page}.png', 'status': status, 'confidence': conf,
               'main': main_, 'right': right, 'left': left, 'note': note,
               'source': str(src.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(src)}
        if status in ('收录', '待核'):
            out = DST / book / f'page_{page}.png' if status == '收录' else DST / '待核' / f'{book}_page_{page}.png'
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
            row['dataset_path'] = str(out.relative_to(ROOT)).replace('\\', '/')
        manifest.append(row)
    with open(DST / '清单.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['book', 'page', 'status', 'confidence', 'main', 'right', 'left',
                                          'note', 'dataset_path', 'source', 'sha256'])
        w.writeheader()
        for r in manifest:
            w.writerow({k: r.get(k, '') for k in w.fieldnames})
    (DST / '清单.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf8')
    books = {}
    for r in manifest:
        if r['status'] == '收录':
            books[r['book']] = books.get(r['book'], 0) + 1
    print('收录', sum(books.values()), books)
    print('待核', sum(1 for r in manifest if r['status'] == '待核'))


if __name__ == '__main__':
    main()
