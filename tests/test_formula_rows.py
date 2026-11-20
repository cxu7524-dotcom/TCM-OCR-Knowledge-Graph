import unittest
from app.schemas import Block
from ocr.normalize import normalize_text
from extract.formula_segments import extract_formula_candidates, _Stream
from extract.formula_rows import extract_formula_rows


def blocks(*texts):
    return [Block(page_id='dev',line_id=f'R{n:03}',block_ref=f'dev:R{n:03}',reading_order=n,col_id=n,
                  bbox=[1000-40*n,200,1030-40*n,700],text=t,text_norm=normalize_text(t),conf=0.95)
            for n,t in enumerate(texts,1)]

def values(g):
    return [(r['herb']['value'],r['dose']['value']) for r in g['rows']]

class TestFormulaRows(unittest.TestCase):
    def test_preparation_boundary_does_not_contaminate_last_dose(self):
        bs = blocks('又一方', '甘草二錢', '不加引水煎溫服')
        group = extract_formula_rows(bs)['formulas'][0]
        self.assertEqual(values(group), [('甘草', '二錢')])
        self.assertEqual(group['composition']['text'], '甘草二錢')

    def test_dose_delimited_regional_prefix_retains_source_spelling(self):
        bs = blocks('又一方', '桂枝一錢淮白朮三錢', '水煎服')
        rows = extract_formula_rows(bs)['formulas'][0]['rows']
        self.assertEqual(values({'rows': rows}), [('桂枝', '一錢'), ('淮白朮', '三錢')])
        self.assertEqual(rows[1]['herb']['source']['text'], '淮白朮')

    def test_distinct_doses_remain_distinct(self):
        bs = blocks('又一方', '甘草一錢桂枝一錢半', '水煎服')
        self.assertEqual(values(extract_formula_rows(bs)['formulas'][0]),
                         [('甘草', '一錢'), ('桂枝', '一錢半')])

    def test_range_dose_remains_complete_and_shared_scope_is_local(self):
        bs = blocks('又一方', '人參自一錢可以加至五錢白朮二錢', '水煎服')
        self.assertEqual(values(extract_formula_rows(bs)['formulas'][0]),
                         [('人參', '自一錢可以加至五錢'), ('白朮', '二錢')])
        bs = blocks('又一方', '人參白朮各二錢茯苓炒三錢', '水煎服')
        self.assertEqual(values(extract_formula_rows(bs)['formulas'][0]),
                         [('人參', '二錢'), ('白朮', '二錢'), ('茯苓', '三錢')])

    def test_source_spelling_and_size_modifier_are_preserved(self):
        bs = blocks('又一方', '大茯苓三錢蒼朮二錢', '水煎服')
        rows = extract_formula_rows(bs)['formulas'][0]['rows']
        self.assertEqual(values({'rows': rows}), [('大茯苓', '三錢'), ('蒼朮', '二錢')])
        self.assertEqual(rows[0]['herb']['source']['text'], '大茯苓')
        self.assertEqual(rows[1]['herb']['source']['text'], '蒼朮')

    def test_page35_two_formulas_and_shared_across_blocks(self):
        # 同页多方；第一方的共享剂量字符在两个相邻 OCR 块中。
        bs=blocks('三化湯','治中風不省人事者服之','羌活枳實厚朴','川大黃各三','錢芒硝一錢半','水煎服',
                  '六君子湯','人參二錢白朮五錢正白苓二錢','甘草廣皮製半夏各一錢','引加生薑水煎服')
        g=extract_formula_rows(bs)['formulas']
        self.assertEqual([x['name']['text'] for x in g],['三化湯','六君子湯'])
        self.assertEqual(values(g[0]),[('羌活','三錢'),('枳實','三錢'),('厚朴','三錢'),('川大黃','三錢'),('芒硝','一錢半')])
        self.assertEqual(values(g[1]),[('人參','二錢'),('白朮','五錢'),('正白苓','二錢'),('甘草','一錢'),('廣皮','一錢'),('製半夏','一錢')])
        for row in g[0]['rows'][:4]:
            self.assertEqual(row['dose']['source']['text'],'各三錢')
            self.assertEqual([s['block_ref'] for s in row['dose']['source']['sources']],['dev:R004','dev:R005'])
        self.assertEqual(g[0]['rows'][0]['herb']['source']['text'],'羌活')
        self.assertNotIn('生薑',str(values(g[1])))

    def test_actual_cross_column_shared_dose_page35(self):
        # 真实版面：药名在一列，末味和“各三錢”在下一列。
        bs=blocks('三化湯','羌活枳實厚朴','川大黃各三錢芒硝一錢半','水煎服')
        rows=extract_formula_rows(bs)['formulas'][0]['rows']
        self.assertEqual([r['dose']['value'] for r in rows[:4]],['三錢']*4)
        self.assertEqual([r['dose']['source']['sources'][0]['block_ref'] for r in rows[:4]],['dev:R003']*4)
        self.assertEqual(rows[0]['herb']['source']['sources'][0]['block_ref'],'dev:R002')

    def test_clear_absence_default_and_manual_verified(self):
        bs=blocks('稀涎散','白礬生用牙皂角','共為細末每服二錢')
        auto=extract_formula_rows(bs)['formulas'][0]
        self.assertEqual(values(auto),[('白礬','原文未载'),('牙皂角','原文未载')])
        self.assertEqual(auto['rows'][0]['dose']['absence_basis'],'自动结构判定')
        self.assertEqual(auto['rows'][0]['dose']['absence_evidence']['terminator']['text'],'共為細末')
        group=extract_formula_rows(bs,verified_complete=True)['formulas'][0]
        self.assertEqual(values(group),[('白礬','原文未载'),('牙皂角','原文未载')])
        self.assertEqual(group['rows'][0]['dose']['evidence_context']['text'],'白礬生用牙皂角')
        self.assertEqual(group['rows'][0]['dose']['absence_basis'],'人工核实')

    def test_absence_cross_block_and_uncertain_conditions(self):
        clean=blocks('稀涎散','白礬生用','牙皂角','共為細末')
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(clean)['formulas'][0]['rows']],['原文未载','原文未载'])
        low=blocks('稀涎散','白礬生用','牙皂角','共為細末');low[2].conf=.89
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(low)['formulas'][0]['rows']],['待核','待核'])
        missing=blocks('稀涎散','白礬生用','牙皂角','共為細末');missing[2].reading_order=4;missing[3].reading_order=5
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(missing)['formulas'][0]['rows']],['待核','待核'])
        damaged=blocks('稀涎散','白礬生用牙皂角','共為細求')
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(damaged)['formulas'][0]['rows']],['待核','待核'])
        tail=blocks('稀涎散','白礬生用牙皂角')
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(tail)['formulas'][0]['rows']],['待核','待核'])
        bad=blocks('稀涎散','白礬〔?〕牙皂角','共為細末')
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(bad)['formulas'][0]['rows']],['待核','待核'])
        altered=blocks('稀涎散','白礬生用牙皂角','共為細末');altered[1].text_norm+=' '
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(altered)['formulas'][0]['rows']],['待核','待核'])
        low_end=blocks('稀涎散','白礬生用牙皂角','共為細末');low_end[-1].conf=.5
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(low_end)['formulas'][0]['rows']],['待核','待核'])

    def test_manual_verified_still_allows_per_row_absence(self):
        bs=blocks('稀涎散','白礬〔?〕牙皂角','共為細末')
        rows=extract_formula_rows(bs,verified_complete=True)['formulas'][0]['rows']
        self.assertEqual([r['dose']['value'] for r in rows],['待核','原文未载'])

    def test_shared_equal_and_dose_variants(self):
        bs=blocks('又一方','陳皮香附吳茱萸良薑石菖蒲各等分','水煎服')
        rows=extract_formula_rows(bs)['formulas'][0]['rows']
        self.assertEqual([r['dose']['value'] for r in rows],['等分']*5)
        self.assertTrue(all(r['dose']['source']['text']=='各等分' for r in rows))
        bs=blocks('又一方','甘草二兩半桂枝少許','水煎服')
        self.assertEqual(values(extract_formula_rows(bs)['formulas'][0]),[('甘草','二兩半'),('桂枝','少許')])

    def test_low_conf_and_explicit_ambiguity(self):
        bs=blocks('桂枝湯','甘草二錢桂枝三錢','水煎服');bs[1].conf=.3
        rows=extract_formula_rows(bs)['formulas'][0]['rows']
        self.assertTrue(all(x['herb']['value']=='待核' and x['dose']['value']=='待核' for x in rows))
        self.assertEqual(rows[0]['dose']['source']['text'],'二錢')
        bs=blocks('桂枝湯','甘草二錢桂枝三錢','水煎服')
        candidate=extract_formula_candidates(bs)['candidates'][0]
        a=candidate['composition']['normalized_range'][0]
        rows=extract_formula_rows(bs,[candidate],uncertain_ranges=[(a+2,a+4)])['formulas'][0]['rows']
        self.assertEqual(rows[0]['dose']['value'],'待核')
        self.assertEqual(rows[0]['dose']['source']['text'],'二錢')
        self.assertEqual(rows[1]['dose']['value'],'三錢')

    def test_dose_gap_uncertain_and_orphan_not_shared_later(self):
        bs=blocks('桂枝湯','桂枝芍藥二錢甘草大棗各三錢','水煎服')
        rows=extract_formula_rows(bs)['formulas'][0]['rows']
        self.assertEqual([r['dose']['value'] for r in rows],['待核','二錢','三錢','三錢'])
        bs=blocks('桂枝湯','桂枝洛三錢甘草二錢','水煎服')
        rows=extract_formula_rows(bs)['formulas'][0]['rows']
        self.assertEqual(rows[0]['dose']['value'],'待核')
        self.assertEqual(rows[0]['dose']['source']['text'],'三錢')

    def test_ocr_ambiguous_tail_and_unreadable_gap(self):
        bs=blocks('三化湯','羌活枳實厚朴','川大黃各三錢芒硝一錢牛','水煎服')
        rows=extract_formula_rows(bs)['formulas'][0]['rows']
        self.assertEqual(rows[-1]['dose']['value'],'待核')
        self.assertEqual(rows[-1]['dose']['source']['text'],'一錢')
        bs=blocks('稀涎散','白礬〔?〕牙皂角','共為細末')
        rows=extract_formula_rows(bs,verified_complete=True)['formulas'][0]['rows']
        self.assertEqual(rows[0]['dose']['value'],'待核')
        self.assertEqual(rows[1]['dose']['value'],'原文未载')

    def test_low_confidence_name_is_not_presented_as_confirmed(self):
        bs=blocks('桂枝湯','桂枝三錢甘草二錢','水煎服');bs[0].conf=.3
        g=extract_formula_rows(bs)['formulas'][0]
        self.assertEqual(g['name_field']['value'],'待核')
        self.assertEqual(g['name_field']['source']['text'],'桂枝湯')

    def test_development_page23_reviewed_range_with_ambiguous_doses(self):
        # 步骤 1 D23-1 人工记录；第 2 步未识别此题名，故以已核对范围构造候选。
        bs=blocks('萬靈九轉還丹秘方','真鴉片冬三兩夏〔?〕石草霜九錢西牛黃〔?〕麝香〔?〕','右為細末')
        st=_Stream(bs);a=st.parts[1]['start'];b=st.parts[1]['end']
        c={'page_id':'dev','name_status':'有名方','name':st.source(0,a),'start_cue':st.source(0,a),
           'composition':st.source(a,b),'passage':st.source(0,b),'end_reason':'制法服法','continuation':'未见续页迹象'}
        g=extract_formula_rows(bs,[c],verified_complete=True)['formulas'][0]
        self.assertEqual(values(g),[('真鴉片','待核'),('石草霜','九錢'),('西牛黃','待核'),('麝香','待核')])
        self.assertIn('夏〔?〕',g['context']['text'])
        self.assertEqual(g['rows'][0]['dose']['source']['text'],'三兩')

    def test_missing_dose_in_unfinished_page_remains_uncertain(self):
        bs=blocks('桂枝湯','桂枝甘草')
        self.assertEqual([r['dose']['value'] for r in extract_formula_rows(bs,verified_complete=True)['formulas'][0]['rows']],['待核','待核'])

if __name__=='__main__':unittest.main()
