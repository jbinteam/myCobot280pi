"""Immutable v3-cache conversion to explicit current-to-target Cartesian residuals."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .common import file_sha256, robot_schema, write_json
from .cartesian_conditioning import CONDITIONING, RELATIVE_CONDITIONING
from ..kinematics import ARM


def _transform_observations(observations, direction, device='cpu', batch_size=32768):
    observations = np.asarray(observations, np.float32)
    if observations.ndim != 2 or observations.shape[1] != 34 or not np.isfinite(observations).all():
        raise ValueError('Expected finite Cartesian observations [N,34]')
    result = observations.copy()
    with torch.no_grad():
        for start in range(0, len(result), batch_size):
            chunk = result[start:start+batch_size]
            q = ARM.center+chunk[:, :6].astype(np.float64)*ARM.radius
            transform = ARM.torch_transform(torch.as_tensor(q, device=device))
            current = torch.cat((transform[:, :3, 3]/ARM.scale, transform[:, :3, 2]), dim=1).cpu().numpy()
            chunk[:, 6:12] += direction*current
    return result


def relative_observations(observations, device='cpu', batch_size=32768):
    return _transform_observations(observations, -1, device, batch_size)


def absolute_observations(observations, device='cpu', batch_size=32768):
    """Reconstruct task Cartesian labels for schema4 diagnostics, never goal joints."""
    return _transform_observations(observations, 1, device, batch_size)


def build_cache(source, augmentation, folder, device='cpu',
                source_cache=Path('simulation/artifacts/diffusion_cartesian_v3/cache'), chunk_frames=32768):
    source_cache = Path(source_cache); folder = Path(folder)
    if source_cache.resolve() == folder.resolve(): raise ValueError('Preserve the source Cartesian cache')
    original = json.loads((source_cache/'cache.json').read_text())
    if original['identity']['conditioning'] != CONDITIONING or original['identity']['robot'] != robot_schema():
        raise ValueError('Expected compatible absolute Cartesian v3 cache')
    source_sha = file_sha256(source); augmentation_sha = file_sha256(augmentation)
    if source_sha != original['identity']['source_sha256'] or augmentation_sha != original['identity']['augmentation_sha256']:
        raise ValueError('Source datasets changed since absolute Cartesian cache creation')
    files = [source_cache/f'{split}_{key}.npy' for split in ('train', 'validation', 'test')
             for key in ('obs', 'q', 'ends', 'ranges_0', 'ranges_1')]
    hashes = {path.name: file_sha256(path) for path in files}
    identity = {'source': str(source), 'source_sha256': source_sha,
                'augmentation': str(augmentation), 'augmentation_sha256': augmentation_sha,
                'robot': robot_schema(), 'conditioning': RELATIVE_CONDITIONING, 'cache_schema': 1,
                'source_cartesian_cache': str(source_cache), 'source_cache_identity': original['identity'],
                'source_cache_manifest_sha256': file_sha256(source_cache/'cache.json'),
                'source_cache_arrays_sha256': hashes,
                'conversion': 'Current normalized q -> shared float64 batched URDF FK; subtract XYZ/scale and directed flange-Z from absolute v3 labels'}
    identity['dataset_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True); manifest = folder/'cache.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['identity'] != identity: raise ValueError('Relative cache sources/schema changed')
        for split in ('train', 'validation', 'test'):
            for key in ('obs', 'q', 'ends', 'ranges_0', 'ranges_1'):
                before = np.load(source_cache/f'{split}_{key}.npy', mmap_mode='r')
                after = np.load(folder/f'{split}_{key}.npy', mmap_mode='r')
                if before.shape != after.shape or before.dtype != after.dtype:
                    raise ValueError('Relative cache is incomplete')
        return saved
    if chunk_frames < 1: raise ValueError('Positive conversion chunk size required')
    for split in ('train', 'validation', 'test'):
        source_obs = np.load(source_cache/f'{split}_obs.npy', mmap_mode='r')
        if source_obs.shape != (original['frames'][split], 34) or source_obs.dtype != np.float32:
            raise ValueError('Incompatible original cache observation array')
        output = np.lib.format.open_memmap(folder/f'{split}_obs.npy', mode='w+', dtype=np.float32, shape=source_obs.shape)
        for start in range(0, len(source_obs), chunk_frames):
            output[start:start+chunk_frames] = relative_observations(source_obs[start:start+chunk_frames], device, chunk_frames)
        output.flush()
        for key in ('q', 'ends', 'ranges_0', 'ranges_1'):
            shutil.copyfile(source_cache/f'{split}_{key}.npy', folder/f'{split}_{key}.npy')
        print(f'Relative Cartesian cache {split}: {len(source_obs)} frames', flush=True)
    saved = {**original, 'identity': identity,
             'target_label_source': 'Immutable absolute v3 Cartesian labels minus current shared URDF FK; no goal joint observations',
             'test_isolation': 'Only train/validation arrays loaded by training; original parent splits/ranges inherited byte-for-byte'}
    write_json(manifest, saved)
    return saved
