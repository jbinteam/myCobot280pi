"""Equivalence and asset-invalidation checks for the CPU hot-path changes."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM, rotation
from simulation import pose5, prm


def original_fk(q):
    t=np.eye(4);origins=[];axes=[];index=0;links={ARM.base_frame:t.copy()}
    for name,(base,axis) in zip(ARM.link_names,ARM.chain):
        t=t@base
        if axis is not None:
            origins.append(t[:3,3].copy());axes.append(t[:3,:3]@axis)
            r=np.eye(4);r[:3,:3]=rotation(axis,q[index]);t=t@r;index+=1
        links[name]=t.copy()
    j=np.stack([np.cross(a,t[:3,3]-p) for a,p in zip(axes,origins)],axis=1)
    return t,np.vstack((j,np.stack(axes,axis=1))),links


def original_axis_error(current,target):
    current=pose5.unit_axis(current);target=pose5.unit_axis(target)
    cross=np.cross(current,target);norm=np.linalg.norm(cross)
    angle=np.arctan2(norm,np.clip(current@target,-1.,1.))
    reference=np.eye(3)[np.argmin(np.abs(current))]
    b1=np.cross(current,reference);b1/=np.linalg.norm(b1)
    basis=np.stack([b1,np.cross(current,b1)])
    vector=cross/norm*angle if norm>1e-10 else (b1*angle if current@target<0 else np.zeros(3))
    return vector,basis,float(np.rad2deg(angle))


class HotPathEquivalenceTests(unittest.TestCase):
    def test_fk_jacobian_and_collision_frames_match_original_at_bounds_and_random_poses(self):
        rng=np.random.default_rng(20261002)
        for q in [ARM.lower,ARM.upper,ARM.center,*rng.uniform(ARM.lower,ARM.upper,(100,6))]:
            expected,j,links=original_fk(q)
            actual,actual_j=ARM.spatial_jacobian(q)
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
            np.testing.assert_allclose(ARM.transform(q),expected,atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual_j,j,atol=1e-12,rtol=0)
            for name,t in ARM.link_transforms(q).items():
                np.testing.assert_allclose(t,links[name],atol=1e-12,rtol=0)

    def test_axis_math_matches_original_including_aligned_and_antipodal_vectors(self):
        rng=np.random.default_rng(20261002)
        pairs=[(rng.normal(size=3),rng.normal(size=3)) for _ in range(100)]
        for axis in ([0.,0.,1.],[1.,0.,0.],[.3,-.8,.5]):
            pairs.extend([(axis,axis),(axis,-np.array(axis)),(np.array(axis)*1e300,np.array(axis)*1e-300)])
        for current,target in pairs:
            expected=original_axis_error(current,target);actual=pose5.axis_error(current,target)
            for a,b in zip(actual,expected):np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)


class GeometryFingerprintTests(unittest.TestCase):
    def fixture(self,directory):
        package=Path(directory)/'package';urdf=package/'urdf/robot/robot.urdf'
        urdf.parent.mkdir(parents=True);(package/'meshes').mkdir()
        mesh=package/'meshes/body.bin';mesh.write_bytes(b'first')
        filename='package://mycobot_description/meshes/body.bin'
        urdf.write_text(f'<robot><mesh filename="{filename}"/></robot>')
        return urdf,mesh,filename

    def test_digest_matches_full_bytes_and_cache_does_not_reread_unchanged_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf,mesh,name=self.fixture(directory)
            expected=hashlib.sha256(urdf.read_bytes());expected.update(name.encode());expected.update(mesh.read_bytes())
            original=Path.read_bytes
            with patch.object(prm,'URDF',urdf),patch.object(Path,'read_bytes',autospec=True,side_effect=original) as read:
                self.assertEqual(prm.geometry_digest(),expected.hexdigest())
                calls=read.call_count
                self.assertEqual(calls,2)
                self.assertEqual(prm.geometry_digest(),expected.hexdigest())
                self.assertEqual(read.call_count,calls)

    def test_same_size_edit_with_restored_mtime_and_deleted_mesh_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf,mesh,_=self.fixture(directory)
            with patch.object(prm,'URDF',urdf):
                old=prm.geometry_digest();stat=mesh.stat()
                mesh.write_bytes(b'other');os.utime(mesh,ns=(stat.st_atime_ns,stat.st_mtime_ns))
                self.assertNotEqual(prm.geometry_digest(),old)
                mesh.unlink()
                with self.assertRaises(FileNotFoundError):prm.geometry_digest()

    def test_unavailable_event_watcher_falls_back_to_fresh_content_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf,mesh,_=self.fixture(directory);original=Path.read_bytes
            with patch.object(prm,'URDF',urdf),patch.object(prm._GEOMETRY_WATCH,'revision',return_value=None), \
                    patch.object(Path,'read_bytes',autospec=True,side_effect=original) as read:
                old=prm.geometry_digest();calls=read.call_count
                self.assertEqual(prm.geometry_digest(),old)
                self.assertEqual(read.call_count,calls+2)
                stat=mesh.stat();mesh.write_bytes(b'other')
                os.utime(mesh,ns=(stat.st_atime_ns,stat.st_mtime_ns))
                self.assertNotEqual(prm.geometry_digest(),old)

    def test_urdf_asset_list_and_symlink_target_changes_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf,mesh,_=self.fixture(directory)
            other=mesh.with_name('other.bin');other.write_bytes(b'other')
            with patch.object(prm,'URDF',urdf):
                old=prm.geometry_digest()
                urdf.write_text(urdf.read_text().replace('body.bin','other.bin'))
                self.assertNotEqual(prm.geometry_digest(),old)
                other.unlink();other.symlink_to(mesh)
                linked=prm.geometry_digest()
                third=mesh.with_name('third.bin');third.write_bytes(b'third')
                other.unlink();other.symlink_to(third)
                self.assertNotEqual(prm.geometry_digest(),linked)

    def test_asset_changed_during_read_is_rejected_and_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf,mesh,_=self.fixture(directory);original=Path.read_bytes
            def changed(path):
                data=original(path)
                if path==mesh:mesh.write_bytes(b'other')
                return data
            with patch.object(prm,'URDF',urdf):
                with patch.object(Path,'read_bytes',autospec=True,side_effect=changed):
                    with self.assertRaisesRegex(ValueError,'during fingerprinting'):prm.geometry_digest()
                self.assertEqual(len(prm.geometry_digest()),64)


if __name__=='__main__':unittest.main()
