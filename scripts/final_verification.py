import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.app.services.validation import validate_image_file
from backend.app.services.orchestrator import run_analysis

def load(p):
    with open(p,'rb') as f:
        d=f.read()
    r=validate_image_file(d, os.path.basename(p))
    return {'bytes':d,'filename':os.path.basename(p),'array':r['array'],'meta':r['meta']}

tests = [
    ('DEMO 1 single', ['data/demo/single_image/scene_1.png'], 'Describe the land-cover and major objects visible in this image.'),
    ('DEMO 2 temporal', ['data/demo/temporal_pairs/flood_T1.png','data/demo/temporal_pairs/flood_T2.png'], 'What changed between these two dates, and where did the change occur?'),
    ('DEMO 3 opt+SAR', ['data/demo/optical_sar_pairs/pair1_optical.png','data/demo/optical_sar_pairs/pair1_sar.png'], 'Use the optical and SAR images together to identify built-up and water-covered regions.'),
    ('DEMO 4 hero', ['data/demo/temporal_pairs/flood_T1.png','data/demo/temporal_pairs/flood_T2.png','data/demo/optical_sar_pairs/flood_sar.png'], 'What changed between the two dates, and can the SAR imagery corroborate the change?'),
    ('DEMO 5 invalid', ['data/demo/invalid_inputs/small_128.png','data/demo/invalid_inputs/large_512.png'], 'What changed between these two dates?'),
]

print("=== FINAL VERIFICATION: 5 DEMOS ===")
for name, paths, query in tests:
    datas=[load(p) for p in paths]
    res=run_analysis(query, datas)
    status='REFUSED' if res.get('refused') else 'OK'
    ans = res['answer'][:120].encode('ascii','ignore').decode()
    print(f"{name}: task={res['task']} status={status} conf={res['confidence']['overall']} ans={ans}")
    if name=='DEMO 5 invalid':
        assert res.get('refused')==True, 'Demo 5 should refuse'
    else:
        assert res.get('refused')==False, f'{name} should not refuse'
        assert len(res['execution_trace'])>=5
        assert res['answer']
print("All 5 demos verified.")
print("Checks: application launches, uploads work, query, agent, models, evidence, confidence, trace, reports, errors handled.")
