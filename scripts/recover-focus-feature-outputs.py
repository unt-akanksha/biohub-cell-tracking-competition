"""Resume this completed feature download, with four bounded HTTP workers."""
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import zipfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'.biohub/cache/kernel-outputs/focus-adaptation-features-v1'


def valid_existing(path):
    if not path.is_file():return False
    try:
        if path.suffix=='.npz':
            with zipfile.ZipFile(path) as archive:return archive.testzip() is None
        if path.suffix=='.json':json.loads(path.read_text(encoding='utf-8'));return True
        if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'));return True
    except (ValueError,zipfile.BadZipFile,SyntaxError,UnicodeError,EOFError):return False
    return False


def main():
    api=KaggleApi();api.authenticate();files=[];token=None
    with api.build_kaggle_client() as client:
        while True:
            request=ApiListKernelSessionOutputRequest();request.user_name='indarkarhana';request.kernel_slug='biohub-focus-adaptation-features-v1';request.page_size=100
            if token:request.page_token=token
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                name=item.file_name
                required=(name=='focus_adaptation_features/launcher_terminal.json'
                    or name.startswith('focus_adaptation_features/outputs/')
                    or (name.startswith('focus_adaptation_features/runtime/') and len(Path(name).parts)==3 and Path(name).suffix in ('.py','.json')))
                if required:files.append(item)
            token=response.next_page_token
            if not token:break
    def fetch(item):
        path=(TARGET/item.file_name).resolve()
        if not path.is_relative_to(TARGET.resolve()):raise ValueError('Output path escaped exact recovery directory')
        if valid_existing(path):return 'retained'
        # URLs may contain credentials: never log them or HTTP exception text.
        for attempt in range(3):
            try:
                response=requests.get(item.url,timeout=(10,60));response.raise_for_status()
                path.parent.mkdir(parents=True,exist_ok=True)
                temporary=path.with_name(path.name+'.download');temporary.write_bytes(response.content);temporary.replace(path)
                if not valid_existing(path):raise ValueError('Invalid downloaded artifact')
                return 'downloaded'
            except (requests.RequestException,ValueError):
                if attempt==2:raise RuntimeError('Bounded output recovery failed for '+item.file_name) from None
    counts={'retained':0,'downloaded':0}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fetch,item) for item in files]
        for n,future in enumerate(as_completed(futures),1):
            counts[future.result()]+=1
            if n%100==0:print(json.dumps(dict(completed=n,total=len(files),**counts)),flush=True)
    print(json.dumps(dict(status='download_complete_not_yet_hash_verified',total=len(files),**counts)),flush=True)


if __name__=='__main__':main()
