import unittest
from app.schemas import Block, TableStruct
from ocr.normalize import normalize_text
from extract.formula_segments import extract_formula_candidates
from extract.formula_rows import extract_formula_rows
from extract.formula_tables import text_formula_tables
from app.main import _tables_payload


def blocks(*texts):
    return [Block(page_id='P',line_id=f'R{i:02}',block_ref=f'P:R{i:02}',reading_order=i,
                  text=t,text_norm=normalize_text(t),conf=.98,bbox=[900-i*40,200,930-i*40,700])
            for i,t in enumerate(texts,1)]

def extract(bs):
    seg=extract_formula_candidates(bs)
    f=extract_formula_rows(bs,seg['candidates'])
    f['continuation_hints']=seg['continuation_hints']
    return f

class TestFormulaApiPayload(unittest.TestCase):
    def test_same_name_separate_origin_retained(self):
        bs=blocks('桂枝湯','桂枝三錢甘草二錢','水煎服','桂枝湯','桂枝四錢甘草二錢','水煎服')
        out=_tables_payload({'page_id':'P','is_demo':False,'tables':[]},[x.to_dict() for x in bs],extract(bs))
        self.assertEqual([t['source_type'] for t in out['tables']],['正文抽取','正文抽取'])
        self.assertNotEqual(out['tables'][0]['table_id'],out['tables'][1]['table_id'])
        self.assertEqual(out['tables'][0]['rows'][1],['桂枝湯','桂枝','三錢'])
        self.assertEqual(out['tables'][1]['rows'][1],['桂枝湯','桂枝','四錢'])
        self.assertNotIn('未识别到方剂表',out['tables_note'])

    def test_distinct_printed_and_text_and_multiref(self):
        bs=blocks('桂枝湯','桂枝各三','錢甘草二錢','水煎服','藥名','劑量','桂枝','三兩')
        t=TableStruct(page_id='P',table_id='P-T01',rows=[['藥名','劑量'],['桂枝','三兩']],
                      cell_refs=[['P:R05','P:R06'],['P:R07','P:R08']])
        a=_tables_payload({'page_id':'P','is_demo':False,'tables':[t.to_dict()]},[b.to_dict() for b in bs],extract(bs))
        self.assertEqual([x['source_type'] for x in a['tables']],['原书表格','正文抽取'])
        f=a['tables'][1]
        self.assertEqual(f['cell_refs_multi'][1][2],['P:R02','P:R03'])
        self.assertEqual(f['cell_refs'][1][2],'P:R02')
        self.assertEqual(f['field_details'][1][2]['source']['text'],'各三錢')
        self.assertEqual(a['duplicate_formula_tables_suppressed'],0)
        self.assertEqual(a,_tables_payload({'page_id':'P','is_demo':False,'tables':[t.to_dict()]},[b.to_dict() for b in bs],extract(bs)))

    def test_duplicate_only_if_all_cells_have_exact_same_origin(self):
        bs=blocks('桂枝湯','桂枝','三錢','甘草','二錢','水煎服')
        data=extract(bs)
        p={'table_id':'T','rows':[['桂枝','三錢'],['甘草','二錢']],
           'cell_refs':[['P:R02','P:R03'],['P:R04','P:R05']]}
        tables,n=text_formula_tables(data,[p],bs,False)
        self.assertEqual(tables,[]);self.assertEqual(n,1)
        p['cell_refs'][0][0]='different'
        tables,n=text_formula_tables(data,[p],bs,False)
        self.assertEqual(len(tables),1);self.assertEqual(n,0)

    def test_truncated_formula_retains_continuation_state(self):
        bs=blocks('桂枝湯','桂枝三錢甘草二錢')
        out=_tables_payload({'page_id':'P','tables':[]},[b.to_dict() for b in bs],extract(bs))
        self.assertEqual(out['tables'][0]['continuation'],'可能续页')
        self.assertEqual(out['tables'][0]['cell_statuses'][1][2],'已识别')

    def test_uncertain_absence_and_legacy(self):
        bs=blocks('稀涎散','白礬生用牙皂角','共為細末');data=extract(bs)
        out=_tables_payload({'page_id':'P','tables':[]},[b.to_dict() for b in bs],data)
        table=out['tables'][0]
        self.assertEqual(table['cell_statuses'][1][2],'原文未载')
        self.assertEqual(table['cell_refs'][1][2],'P:R02')
        self.assertEqual(table['cell_ref_scopes'][1][2],'配伍上下文')
        self.assertEqual(table['field_details'][1][2]['absence_basis'],'自动结构判定')
        bs[1].conf=.2;data=extract(bs)
        table=_tables_payload({'page_id':'P','tables':[]},[b.to_dict() for b in bs],data)['tables'][0]
        self.assertEqual(table['cell_statuses'][1][2],'待核')
        self.assertEqual(_tables_payload({'page_id':'old'},[])['tables_status'],'no_table_field')

if __name__=='__main__':unittest.main()
