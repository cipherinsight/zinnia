"""Run from this batch only; final Z# files are authoritative."""
import subprocess,sys,pathlib,os
root=pathlib.Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TMPDIR']=str(root/'tmp')
for script in ['run.py','classify.py','compare.py','report.py']:
 subprocess.run([sys.executable,str(root/script)],cwd=root,check=True)
