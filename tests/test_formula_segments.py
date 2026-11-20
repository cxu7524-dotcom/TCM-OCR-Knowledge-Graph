# -*- coding: utf-8 -*-
import unittest
from app.schemas import Block
from ocr.normalize import normalize_text
from extract.formula_segments import extract_formula_candidates


def blocks(*texts):
    return [Block(page_id='dev', line_id=f'R{n:03}', block_ref=f'dev:R{n:03}',
                  reading_order=n, col_id=n, bbox=[1000-n*40, 200, 1030-n*40, 700],
                  text=text, text_norm=normalize_text(text))
            for n, text in enumerate(texts, 1)]


class TestFormulaSegments(unittest.TestCase):
    def test_preparation_and_derivative_are_not_formulas(self):
        self.assertEqual(extract_formula_candidates(blocks(
            '不加引水煎溫服', '藥理文字白朮茯苓調理脾胃'))['candidates'], [])
        for prefix in ('即', '卽'):
            result = extract_formula_candidates(blocks(
                '桂枝湯', prefix + '前方加桂枝三錢甘草二錢', '白朮二錢茯苓三錢', '水煎服'))
            self.assertEqual(result['candidates'], [])

    def test_named_tail_provides_source_hint_without_fabricating_rows(self):
        result = extract_formula_candidates(blocks('桂枝湯', '治風寒者'))
        self.assertEqual(result['candidates'], [])
        self.assertEqual(len(result['continuation_hints']), 1)
        self.assertEqual(result['continuation_hints'][0]['passage']['text'], '桂枝湯治風寒者')

    def test_dev_danfang22_two_real_transcriptions(self):
        # 步骤 1 D22-1 / D22-2 原图人工转写，以病证标题开头而无方名。
        bs=blocks('心痛 山梔子七枚炒焦黃 水一盞煎七分 入姜汁三匙熱服立止',
                  '寒痛 陳皮香附吳茱萸良薑石菖蒲各等分 水一碗')
        r=extract_formula_candidates(bs)
        self.assertEqual(len(r['candidates']),2)
        a,b=r['candidates']
        self.assertEqual([a['name_status'],b['name_status']],['无名方','无名方'])
        self.assertEqual([a['start_cue']['text'],b['start_cue']['text']],['心痛','寒痛'])
        self.assertEqual(a['composition']['text'],'山梔子七枚炒焦黃 ')
        self.assertEqual(b['composition']['text'],'陳皮香附吳茱萸良薑石菖蒲各等分 ')
        self.assertEqual(a['composition']['sources'][0]['block_ref'],'dev:R001')
        self.assertEqual(b['composition']['sources'][0]['block_ref'],'dev:R002')
        self.assertEqual(a['end_reason'],'制法服法')

    def test_explicit_one_herb_and_dose_with_title(self):
        for header in ('桂枝湯','又一方'):
            with self.subTest(header=header):
                r=extract_formula_candidates(blocks(header,'桂枝三錢','水煎服'))
                self.assertEqual(len(r['candidates']),1)
                self.assertEqual(r['candidates'][0]['composition']['text'],'桂枝三錢')
        r=extract_formula_candidates(blocks('又一方','石燕三錢','水煎服'))
        self.assertEqual(r['candidates'][0]['composition']['text'],'石燕三錢')

    def test_named_single_in_top_aligned_same_block(self):
        bs=blocks('桂枝湯桂枝三錢','水煎服','以下是另外的說明文字')
        bs[0].bbox=[500,200,540,700];bs[2].bbox=[400,240,440,700]
        r=extract_formula_candidates(bs)
        self.assertEqual(len(r['candidates']),1)
        self.assertEqual(r['candidates'][0]['name']['text'],'桂枝湯')
        self.assertEqual(r['candidates'][0]['composition']['text'],'桂枝三錢')

    def test_single_herb_negative_contexts(self):
        cases=[('桂枝湯','桂枝性溫主治諸病','水煎服'),
               ('桂枝湯','方論桂枝三錢有溫通之效','水煎服'),
               ('桂枝湯','桂枝三錢性溫主治諸病','水煎服'),
               ('桂枝湯','即前方加桂枝三錢','水煎服'),
               ('桂枝湯','每服桂枝三錢','水煎服'),
               ('心痛者宜用桂枝三錢','水煎服'),
               ('心痛加桂枝三錢','水煎服'),
               ('桂枝三錢','水煎服')]
        for texts in cases:
            with self.subTest(texts=texts):
                self.assertEqual(extract_formula_candidates(blocks(*texts))['candidates'],[])

    def test_preparation_is_not_a_title(self):
        r=extract_formula_candidates(blocks('共搞碎水煎温服', '按此藥雖繁雜實有至理',
            '川芎和肝血也而甘草白术調脾氣也', '稀涎散', '白礬生用牙皂角', '共為細末'))
        self.assertEqual([c['name']['text'] for c in r['candidates']], ['稀涎散'])

    def test_development_page35_multiformula_cross_column(self):
        # 源于步骤 1 page_35 人工转写；按实际竖列拆块。
        bs = blocks('三化湯', '治中風不省人事痰涎壅盛者服之',
                    '羌活枳實厚朴', '川大黃各三錢芒硝一錢半', '水三鍾煎八分溫服',
                    '六君子湯', '治面目萎黃無力者服之', '人參二錢白朮五錢正白苓二錢',
                    '甘草廣皮製半夏各一錢', '引加生薑一錢半紅棗肉三枚水煎溫服')
        result = extract_formula_candidates(reversed(bs))
        self.assertEqual([c['name']['text'] for c in result['candidates']], ['三化湯','六君子湯'])
        first, second = result['candidates']
        self.assertEqual(first['composition']['text'],'羌活枳實厚朴川大黃各三錢芒硝一錢半')
        self.assertEqual(second['composition']['text'],'人參二錢白朮五錢正白苓二錢甘草廣皮製半夏各一錢')
        self.assertEqual([s['block_ref'] for s in first['composition']['sources']], ['dev:R003','dev:R004'])
        self.assertEqual(first['end_reason'],'制法服法')
        self.assertNotIn('引加',second['composition']['text'])

    def test_top_aligned_heading_and_composition_in_same_block(self):
        bs=blocks('三化湯羌活枳實厚朴川大黃各三錢', '水煎溫服', '此後是另外的論述文字而非方名')
        bs[0].bbox=[500,200,540,700];bs[2].bbox=[400,240,440,700]
        r=extract_formula_candidates(bs)
        self.assertEqual(len(r['candidates']),1)
        self.assertEqual(r['candidates'][0]['name']['text'],'三化湯')
        self.assertEqual(r['candidates'][0]['composition']['text'],'羌活枳實厚朴川大黃各三錢')

    def test_name_split_between_adjacent_blocks(self):
        r=extract_formula_candidates(blocks('三化', '湯', '羌活枳實厚朴', '川大黃各三錢芒硝一錢半', '水煎服'))
        self.assertEqual(len(r['candidates']),1)
        c=r['candidates'][0]
        self.assertEqual(c['name']['text'], '三化湯')
        self.assertEqual([x['block_ref'] for x in c['name']['sources']], ['dev:R001','dev:R002'])

    def test_development_page33_no_dose(self):
        r=extract_formula_candidates(blocks('稀涎散','白礬生用牙皂角','共為細末每服二錢'))
        self.assertEqual(len(r['candidates']),1)
        self.assertEqual(r['candidates'][0]['composition']['text'],'白礬生用牙皂角')

    def test_development_page32_and_mention_negative(self):
        r=extract_formula_candidates(blocks('先天真陽不足者宜人參附子湯','通關散',
            '治一切閉證能通關開竅','天南星牙皂角生半夏各二錢','細辛一錢冰片三分','共為細末'))
        self.assertEqual([c['name']['text'] for c in r['candidates']],['通關散'])
        self.assertEqual(r['candidates'][0]['composition']['text'],'天南星牙皂角生半夏各二錢細辛一錢冰片三分')

    def test_development_page36_reference_not_formula(self):
        r=extract_formula_candidates(blocks('去廣皮半夏名四君子湯治脾虛無痰者更效',
            '烏藥順氣湯','治男婦風氣攻注四肢骨節疼痛','正嫩烏藥二錢廣皮一錢半麻黃',
            '川芎殭蠶枳殼','桔梗各一錢乾薑五分甘草三分','引加生薑一錢紅棗肉二枚水煎溫服'))
        self.assertEqual([c['name']['text'] for c in r['candidates']],['烏藥順氣湯'])
        self.assertNotIn('引加',r['candidates'][0]['composition']['text'])

    def test_anonymous_and_cross_page(self):
        r=extract_formula_candidates(blocks('又一方','桂枝三錢','甘草二錢'))
        self.assertEqual(r['candidates'][0]['name_status'],'无名方')
        self.assertEqual(r['candidates'][0]['continuation'],'可能续页')
        self.assertIsNone(r['candidates'][0]['name'])
        h=extract_formula_candidates(blocks('甘草一錢桂枝二錢','水煎服','某湯','治咳嗽'))
        self.assertEqual(len(h['continuation_hints']),1)
        self.assertEqual(h['candidates'],[])

    def test_mention_and_reference_are_not_new_formula(self):
        self.assertEqual(extract_formula_candidates(blocks('宜用桂枝湯加甘草一錢芍藥二錢'))['candidates'],[])
        self.assertEqual(extract_formula_candidates(blocks('桂枝湯','即前方加甘草一錢芍藥二錢'))['candidates'],[])

    def test_page_head_after_short_header(self):
        r=extract_formula_candidates(blocks('思濟堂方書','朴硝連翹甘草各一錢', '水煎服',
            '稀涎散','白礬生用牙皂角','共為細末'))
        self.assertEqual(len(r['continuation_hints']),1)
        self.assertEqual(r['continuation_hints'][0]['passage']['text'],'朴硝連翹甘草各一錢')
        self.assertEqual(r['candidates'][0]['name']['text'],'稀涎散')

    def test_mapping_length_changes_and_insertions_conservative(self):
        bs=blocks('桂枝湯','桂枝三錢甘草二錢','水煎服')
        bs[1].text='桂枝三錢甘草二錢'; bs[1].text_norm='桂枝三錢甘草二錢额'
        c=extract_formula_candidates(bs)['candidates'][0]['composition']
        self.assertEqual(c['text'],bs[1].text)
        self.assertEqual(c['sources'][0]['block_raw_range'],[0,len(bs[1].text)])
        self.assertEqual(c['sources'][0]['mapping_precision'],'block')

    def test_reject_mixed_pages(self):
        bs=blocks('桂枝湯','桂枝三錢甘草二錢');bs[1].page_id='other'
        with self.assertRaises(ValueError):extract_formula_candidates(bs)

if __name__=='__main__':unittest.main()
