"""Check the installed tests and full CLI outputs outside the checkout.

Use --section tests or --section commands to run either half on its own.
The default runs both. These are same-runtime packaging comparisons, not
independent numerical or statistical evidence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
MODULES = ['graphlab', 'graphlab.__main__', 'graphlab.cli', 'graphlab.core',
           'graphlab.demo', 'graphlab.attribution', 'graphlab.edge_audit', 'graphlab.joint']
COMMANDS = [
    ['demo'], ['attribution-demo'], ['edge-demo'], ['joint-demo'],
    ['train', 'train.json', 'trained.json', '--updates', '2'],
    ['predict', 'model.json', 'predict.json'],
    ['explain', 'model.json', 'predict.json', '--node', '0', '--positive', '1', '--negative', '0'],
    ['edges', 'model.json', 'predict.json', '--node', '0', '--positive', '1', '--negative', '0'],
    ['joint', 'model.json', 'predict.json', '--node', '0', '--positive', '1', '--negative', '0', '--max-forwards', '16'],
]


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--section', choices=['all', 'tests', 'commands'], default='all')
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels), str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'docs']:
        shutil.copytree(ROOT / name, outside / name)
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:
    if name != 'graphlab.__main__':importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
if sys.argv[3]=='tests':
    tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
    result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.testsRun>0 and result.wasSuccessful()
    print(json.dumps(dict(tests=result.testsRun)))
check()
print(json.dumps(dict(installed_origins=len(names),checked_before_and_after=True)))
'''
    if args.section in ['all', 'tests']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'tests'], outside))
    if args.section in ['all', 'commands']:
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        reference = work / 'reference'
        reference.mkdir()
        source_env = ENV.copy()
        source_env['PYTHONPATH'] = str(ROOT)
        fixture = json.loads(call([sys.executable, '-m', 'graphlab', 'joint-demo'], reference, source_env))
        for target in [reference, outside]:
            (target / 'model.json').write_text(json.dumps(fixture['model']))
            (target / 'predict.json').write_text(json.dumps(dict(features=fixture['features'], edges=fixture['edges'])))
            (target / 'train.json').write_text(json.dumps(dict(features=[[1.], [2.]], edges=[[0, 1]], labels=[0, 1], train_indices=[0, 1], classes=2)))
        expected = []
        for command in COMMANDS:
            output = call([sys.executable, '-m', 'graphlab'] + command, reference, source_env).strip()
            expected.append(output)
        fresh = json.loads((reference / 'trained.json').read_text())
        for entry in [[py, '-I', '-m', 'graphlab'], [str(work / 'env/bin/graphlab')]]:
            for command, report in zip(COMMANDS, expected):
                output = call(entry + command, outside).strip()
                assert output == report, command
            assert json.loads((outside / 'trained.json').read_text()) == fresh
        print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT), 'imports'], outside))
        print(json.dumps(dict(command_invocations=len(COMMANDS) * 2, command_kinds=9,
                              complete_output_files=1, entrypoints=2)))
