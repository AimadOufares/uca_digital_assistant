import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ['DJANGO_SETTINGS_MODULE'] = 'core.settings'
sys.path.insert(0, os.getcwd())

from rag_module.pipeline import ask_question

print("=" * 55)
print("TEST EN DIRECT AVEC LM STUDIO")
print("=" * 55)
print("Question : Qu'est-ce que la plateforme CIP ?")
print("-" * 55)

t0 = time.time()
result = ask_question("Qu'est-ce que la plateforme CIP ?")
t1 = time.time()

answer = result.get('answer', '')
sources = result.get('sources', [])
source_names = [s.get('name', '?') for s in sources]

print(f"Temps total  : {t1 - t0:.1f} sec")
print(f"Sources      : {source_names}")
print(f"Reponse      :\n{answer[:600]}")
print("=" * 55)
