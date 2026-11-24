"""Pinned platform/GPU-specific bootstrap for the optional temporal extension.

Verified release bytes only: a prebuilt binary for the GPU architecture, or, when
none is published and a CUDA toolkit (nvcc) is installed, a local build from source
files pinned by SHA-256 at the same commit. No pip, site-packages writes or Kitchen
replacement. Build/ABI checks do not imply GPU runtime validation.
"""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
try:
    from .minimax_h3_temporal_lock import file_lock
except ImportError:
    from minimax_h3_temporal_lock import file_lock

MANIFEST = json.loads((Path(__file__).parent/'h3_temporal_binaries.json').read_text())
SOURCE_COMMIT = MANIFEST['source_commit']
RAW = f'https://raw.githubusercontent.com/alisson-anjos/comfy-kitchen/{SOURCE_COMMIT}'
RELEASE = 'https://github.com/alisson-anjos/comfy-kitchen/releases/download/'+MANIFEST['release_tag']
COMMON_ASSETS = (
    ('temporal_attention.py', RAW+'/comfy_kitchen/temporal_attention.py', '756171846469864702903d19e86a5f2b793a1e01b8fab57a72c68306dd1f2ee5', 256*1024),
    ('LICENSE', RAW+'/LICENSE', 'cd1c9ea1573c9047e9b5a3676ecc1c8f755e56f249ac1ebd8b5ceda7899037da', 64*1024),
    ('NOTICE', RAW+'/NOTICE', '35c43127ad3acbd6bc94ff63c56684a7c26eedf07795e596c8105a0476dae2ea', 64*1024),
)
_LOCK = threading.Lock()
_READY = {}
_FAILED = set()


def select_binary(system, machine, capability):
    if system not in ('Linux','Windows') or machine.lower() not in ('x86_64','amd64'):
        return None
    arch = int(capability[0])*10+int(capability[1])
    return next((b for b in MANIFEST['binaries'] if b['platform']==system and b['arch']==arch),None)


def binary_for_device(device):
    import torch
    return select_binary(platform.system(),platform.machine(),torch.cuda.get_device_capability(device))


def cache_base():
    explicit=os.environ.get('TARA_H3_TEMPORAL_CACHE_DIR')
    if explicit:return Path(explicit).expanduser()
    try:
        import folder_paths
        return Path(folder_paths.get_user_directory())/'bfs'/'h3_temporal_kernels'
    except (ImportError,AttributeError):
        return Path.home()/'.cache'/'bfsnodes'/'h3_temporal_kernels'


def cache_root(target):
    return cache_base()/(f"v1-{target['platform'].lower()}-x86_64-sm{target['arch']}-"+target['sha256'][:12])


def driver_version():
    driver=ctypes.WinDLL('nvcuda.dll') if platform.system()=='Windows' else ctypes.CDLL('libcuda.so.1')
    version=ctypes.c_int()
    if driver.cuDriverGetVersion(ctypes.byref(version))!=0:
        raise RuntimeError('Cannot query NVIDIA CUDA driver version')
    return version.value


def compatible(device):
    if os.environ.get('TARA_H3_TEMPORAL_AUTO_INSTALL','1').lower() in ('0','false','off'):
        return False,'automatic setup disabled'
    if platform.system() not in ('Linux','Windows') or platform.machine().lower() not in ('x86_64','amd64') or ctypes.sizeof(ctypes.c_void_p)!=8:
        return False,'automatic setup requires 64-bit Linux or Windows on x86_64'
    import torch
    if getattr(torch.version,'hip',None) or not torch.cuda.is_available():
        return False,'automatic setup requires NVIDIA CUDA'
    target=binary_for_device(device)
    # The sidecar uses CUDA 13's static runtime; PyTorch itself may use an older CUDA.
    if target is not None and driver_version()<MANIFEST['minimum_cuda_driver_version']:
        return False,'the prebuilt requires an NVIDIA driver supporting CUDA 13'
    import comfy_kitchen
    if not comfy_kitchen.int8_attention_is_available(device):
        return False,'Kitchen native INT8 attention unavailable'
    from comfy_kitchen.sage_attention import prequantize_int8_attention
    if not callable(prequantize_int8_attention):
        return False,'Kitchen prequantization API unavailable'
    return True,None


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def download(path,url,expected,limit):
    if path.is_file() and digest(path)==expected:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    temp=Path(name)
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'TARA-H3-Temporal/1'})
        h=hashlib.sha256();total=0
        with os.fdopen(fd,'wb') as out,urllib.request.urlopen(request,timeout=20) as response:
            while True:
                block=response.read(256*1024)
                if not block:break
                total+=len(block)
                if total>limit:raise RuntimeError('Temporal asset exceeds the pinned size limit')
                h.update(block);out.write(block)
        if h.hexdigest()!=expected:raise RuntimeError('Temporal asset checksum mismatch; downloaded code was not loaded')
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def assets_for(target):
    binary=('backends/cuda/native_temporal/'+target['file'],RELEASE+'/'+target['file'],target['sha256'],target['bytes'])
    return (*COMMON_ASSETS,binary)


def provision(root,target):
    with file_lock(root/'.install.lock',60):
        assets=assets_for(target)
        missing=any(not (root/rel).is_file() or digest(root/rel)!=sha for rel,_,sha,_ in assets)
        if missing:
            print(f"[TARA-H3] First-use setup: downloading verified {target['platform']} SM{target['arch']} one-pass extension ({target['bytes']/1e6:.1f} MB). No compilation or pip install.",flush=True)
        # Verify the binary before importing any downloaded Python code.
        for rel,url,sha,limit in sorted(assets,key=lambda a:a[0].endswith('.py')):
            download(root/rel,url,sha,limit)
        return load_api(root,target)


def load_api(root,target):
    binary=root/'backends/cuda/native_temporal'/target['file']
    lib=ctypes.CDLL(str(binary))
    lib.ck_temporal_abi_version.restype=ctypes.c_int;lib.ck_temporal_arch.restype=ctypes.c_int
    if lib.ck_temporal_abi_version()!=target['native_abi'] or lib.ck_temporal_arch()!=target['arch']:
        raise RuntimeError('Temporal native ABI/architecture mismatch')
    name=f"_bfs_ck_temporal_v1_sm{target['arch']}_"+COMMON_ASSETS[0][2][:12]+'_'+hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,root/'temporal_attention.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
        if getattr(module,'VERSION',None)!='h3_native_temporal_kernel_v1':raise RuntimeError('Temporal API version mismatch')
    except BaseException:
        sys.modules.pop(name,None);raise
    print(f"[TARA-H3] Verified {target['platform']} SM{target['arch']} one-pass extension ready: {root}",flush=True)
    return module


def local_toolchain():
    """(nvcc, host compiler or None) for a local build, or a reason string."""
    if os.environ.get('TARA_H3_TEMPORAL_LOCAL_BUILD','1').lower() in ('0','false','off'):
        return 'local build disabled (TARA_H3_TEMPORAL_LOCAL_BUILD=0)'
    if not MANIFEST.get('sources'):
        return 'no pinned sources for a local build'
    home=os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    exe='nvcc.exe' if platform.system()=='Windows' else 'nvcc'
    nvcc=str(Path(home)/'bin'/exe) if home and (Path(home)/'bin'/exe).is_file() else shutil.which('nvcc')
    if not nvcc:
        return 'no prebuilt for this GPU and no CUDA toolkit (nvcc) for a local build'
    compiler=None
    if platform.system()=='Windows':
        compiler=shutil.which('cl')
        if not compiler:
            return 'no prebuilt for this GPU; a local build needs the MSVC compiler (run ComfyUI from a "x64 Native Tools" prompt)'
    return nvcc,compiler


def build_locally(device):
    """Build the extension for this GPU from the pinned sources into the cache; returns the loaded API."""
    import torch
    major,minor=torch.cuda.get_device_capability(device);arch=major*10+minor
    if arch<80:raise RuntimeError(f'SM{arch} is not supported (needs SM80 or newer)')
    tools=local_toolchain()
    if isinstance(tools,str):raise RuntimeError(tools)
    nvcc,compiler=tools
    system=platform.system()
    target=dict(platform=system,arch=arch,native_abi=1,file=f"libck_temporal_sm{arch}"+('.dll' if system=='Windows' else '.so'))
    root=cache_base()/f"v1-{system.lower()}-x86_64-sm{arch}-local-{SOURCE_COMMIT[:12]}"
    with file_lock(root/'.install.lock',1800):
        binary=root/'backends/cuda/native_temporal'/target['file']
        if not binary.is_file():
            print(f'[TARA-H3] No prebuilt for {system} SM{arch}: building the one-pass extension locally with {nvcc} (one time, a few minutes).',flush=True)
            src=root/'src'
            for item in MANIFEST['sources']:
                download(src/item['path'],RAW+'/'+item['path'],item['sha256'],item['bytes'])
            script=src/'comfy_kitchen/backends/cuda/native_temporal/build.py'
            command=[sys.executable,str(script),'--arch',str(arch),'--jobs','2','--build-dir',str(root/'obj'),'--output',str(binary),'--nvcc',nvcc]
            if compiler:command+=['--compiler',compiler]
            binary.parent.mkdir(parents=True,exist_ok=True)
            result=subprocess.run(command,capture_output=True,text=True,timeout=1800)
            if result.returncode!=0:
                raise RuntimeError('local build failed: '+(result.stderr or result.stdout).strip()[-600:])
        for rel,url,sha,limit in COMMON_ASSETS:
            download(root/rel,url,sha,limit)
        return load_api(root,target)


def ensure(device):
    """At most one automatic setup attempt per device/cache location per process."""
    key=(str(cache_base()),platform.system(),str(device))
    with _LOCK:
        if key in _READY:return _READY[key]
        if key in _FAILED:return None
        try:
            ok,reason=compatible(device)
            if not ok:
                _FAILED.add(key)
                print(f'[TARA-H3] Automatic native setup skipped: {reason}; retaining the existing backend.',flush=True)
                return None
            target=binary_for_device(device)
            module=provision(cache_root(target),target) if target is not None else build_locally(device)
            if not module.is_available(device):raise RuntimeError('Verified native extension is unavailable for this device')
            _READY[key]=module
            return module
        except Exception as exc:
            if type(exc).__name__=='InterruptProcessingException':raise
            _FAILED.add(key)
            print(f'[TARA-H3] Automatic native setup unavailable: {exc}. Retaining the existing backend. Restart ComfyUI to retry, or install the extension manually.',flush=True)
            return None
