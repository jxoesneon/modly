"""Tests for api/metal_mesh/__init__.py — Metal 3.0 mesh shader pipeline."""
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

from metal_mesh import (
    is_mesh_shader_supported, build_meshlets, MeshShaderRenderer,
)
import metal_mesh


class IsMeshShaderSupportedTests(unittest.TestCase):
    def test_false_on_non_darwin(self) -> None:
        with patch.object(sys, 'platform', 'linux'):
            self.assertFalse(is_mesh_shader_supported())

    def test_false_on_windows(self) -> None:
        with patch.object(sys, 'platform', 'win32'):
            self.assertFalse(is_mesh_shader_supported())

    def test_false_on_darwin_old_macos(self) -> None:
        with patch.object(sys, 'platform', 'darwin'), \
             patch('metal_mesh.platform.mac_ver', return_value=("12.0", "", "")):
            self.assertFalse(is_mesh_shader_supported())

    def test_true_for_m1_cpufamily(self) -> None:
        proc = MagicMock(returncode=0, stdout=f"{0x1b588bb3}\n")
        with patch.object(sys, 'platform', 'darwin'), \
             patch('metal_mesh.platform.mac_ver', return_value=("13.0", "", "")), \
             patch('metal_mesh.subprocess.run', return_value=proc):
            self.assertTrue(is_mesh_shader_supported())

    def test_true_for_m4_donan_cpufamily(self) -> None:
        proc = MagicMock(returncode=0, stdout=f"{0x6f5129ac}\n")
        with patch.object(sys, 'platform', 'darwin'), \
             patch('metal_mesh.platform.mac_ver', return_value=("13.0", "", "")), \
             patch('metal_mesh.subprocess.run', return_value=proc):
            self.assertTrue(is_mesh_shader_supported())

    def test_false_for_unknown_cpufamily(self) -> None:
        proc = MagicMock(returncode=0, stdout="1\n")
        with patch.object(sys, 'platform', 'darwin'), \
             patch('metal_mesh.platform.mac_ver', return_value=("13.0", "", "")), \
             patch('metal_mesh.subprocess.run', return_value=proc):
            self.assertFalse(is_mesh_shader_supported())

    def test_false_when_sysctl_fails(self) -> None:
        proc = MagicMock(returncode=1, stdout="")
        with patch.object(sys, 'platform', 'darwin'), \
             patch('metal_mesh.platform.mac_ver', return_value=("13.0", "", "")), \
             patch('metal_mesh.subprocess.run', return_value=proc):
            self.assertFalse(is_mesh_shader_supported())


class BuildMeshletsTests(unittest.TestCase):
    def _square_mesh(self) -> tuple:
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        return vertices, faces

    def test_handles_empty_mesh(self) -> None:
        vertices = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.uint32)
        meshlets, indices = build_meshlets(vertices, faces)
        self.assertEqual(meshlets, [])
        self.assertEqual(indices, [])

    def test_handles_single_triangle(self) -> None:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        meshlets, indices = build_meshlets(vertices, faces)
        self.assertEqual(len(meshlets), 1)
        self.assertEqual(meshlets[0]["triangleCount"], 1)
        self.assertEqual(meshlets[0]["vertexCount"], 3)
        self.assertEqual(len(indices), 1)

    def test_partitions_triangles_into_meshlets(self) -> None:
        # 3 triangles with max_triangles=1 => 3 meshlets
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [2, 0, 0], [3, 0, 0], [2, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 2]], dtype=np.uint32)
        meshlets, indices = build_meshlets(vertices, faces, max_triangles=1)
        self.assertEqual(len(meshlets), 3)
        for m in meshlets:
            self.assertEqual(m["triangleCount"], 1)

    def test_respects_max_triangles_limit(self) -> None:
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [2, 0, 0], [3, 0, 0], [2, 1, 0],
            [4, 0, 0], [5, 0, 0], [4, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint32)
        meshlets, _ = build_meshlets(vertices, faces, max_triangles=2)
        # 3 triangles / 2 per meshlet => 2 meshlets
        self.assertEqual(len(meshlets), 2)
        self.assertEqual(meshlets[0]["triangleCount"], 2)
        self.assertEqual(meshlets[1]["triangleCount"], 1)

    def test_respects_max_vertices_limit(self) -> None:
        # Each triangle uses unique vertices so vertex count grows.
        vertices = np.array([
            [i, 0, 0] for i in range(9)
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint32)
        meshlets, _ = build_meshlets(vertices, faces, max_vertices=3, max_triangles=1)
        for m in meshlets:
            self.assertLessEqual(m["vertexCount"], 3)

    def test_computes_bounding_sphere_center(self) -> None:
        vertices, faces = self._square_mesh()
        meshlets, _ = build_meshlets(vertices, faces, max_triangles=2)
        center = np.array(meshlets[0]["center"])
        expected_center = vertices.mean(axis=0)
        np.testing.assert_allclose(center, expected_center, atol=1e-5)

    def test_computes_bounding_sphere_radius(self) -> None:
        vertices, faces = self._square_mesh()
        meshlets, _ = build_meshlets(vertices, faces, max_triangles=2)
        center = np.array(meshlets[0]["center"])
        expected_radius = np.max(np.linalg.norm(vertices - center, axis=1))
        self.assertAlmostEqual(meshlets[0]["radius"], float(expected_radius), places=4)

    def test_meshlet_indices_store_global_vertex_ids(self) -> None:
        vertices, faces = self._square_mesh()
        _, indices = build_meshlets(vertices, faces, max_triangles=2)
        self.assertEqual(len(indices), 2)
        self.assertEqual(indices[0]["v0"], 0)
        self.assertEqual(indices[0]["v1"], 1)
        self.assertEqual(indices[0]["v2"], 2)


class MeshShaderRendererInitTests(unittest.TestCase):
    def test_raises_runtime_error_on_unsupported(self) -> None:
        with patch('metal_mesh.is_mesh_shader_supported', return_value=False):
            with self.assertRaises(RuntimeError):
                MeshShaderRenderer()

    def test_init_sets_none_fields_on_supported(self) -> None:
        with patch('metal_mesh.is_mesh_shader_supported', return_value=True):
            r = MeshShaderRenderer()
        self.assertIsNone(r._vertices)
        self.assertIsNone(r._faces)
        self.assertIsNone(r._meshlets)


class MeshShaderRendererLoadMeshTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = patch('metal_mesh.is_mesh_shader_supported', return_value=True)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()

    def test_load_mesh_stores_vertices_and_faces(self) -> None:
        r = MeshShaderRenderer()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        r.load_mesh(vertices=vertices, faces=faces)
        np.testing.assert_array_equal(r._vertices, vertices.astype(np.float32))
        np.testing.assert_array_equal(r._faces, faces.astype(np.uint32))

    def test_load_mesh_creates_zero_uvs_when_not_provided(self) -> None:
        r = MeshShaderRenderer()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        r.load_mesh(vertices=vertices, faces=faces)
        self.assertEqual(r._uvs.shape, (3, 2))
        self.assertTrue(np.all(r._uvs == 0))

    def test_load_mesh_computes_normals_when_not_provided(self) -> None:
        r = MeshShaderRenderer()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        r.load_mesh(vertices=vertices, faces=faces)
        self.assertIsNotNone(r._normals)
        self.assertEqual(r._normals.shape, (3, 3))

    def test_load_mesh_uses_provided_normals(self) -> None:
        r = MeshShaderRenderer()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        r.load_mesh(vertices=vertices, normals=normals, faces=faces)
        np.testing.assert_array_equal(r._normals, normals.astype(np.float32))

    def test_load_mesh_uses_provided_uvs(self) -> None:
        r = MeshShaderRenderer()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        r.load_mesh(vertices=vertices, uvs=uvs, faces=faces)
        np.testing.assert_array_equal(r._uvs, uvs.astype(np.float32))

    def test_load_mesh_builds_meshlets_when_not_provided(self) -> None:
        r = MeshShaderRenderer()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        r.load_mesh(vertices=vertices, faces=faces)
        self.assertIsNotNone(r._meshlets)
        self.assertIsNotNone(r._meshlet_indices)
        self.assertEqual(len(r._meshlets), 1)

    def test_load_mesh_uses_provided_meshlets(self) -> None:
        r = MeshShaderRenderer()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        meshlets = [{"vertexOffset": 0, "triangleOffset": 0, "vertexCount": 3,
                     "triangleCount": 1, "center": [0, 0, 0], "radius": 1.0}]
        meshlet_indices = [{"v0": 0, "v1": 1, "v2": 2}]
        r.load_mesh(vertices=vertices, faces=faces,
                    meshlets=meshlets, meshlet_indices=meshlet_indices)
        self.assertEqual(r._meshlets, meshlets)
        self.assertEqual(r._meshlet_indices, meshlet_indices)


class MeshShaderRendererRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patch = patch('metal_mesh.is_mesh_shader_supported', return_value=True)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()

    def test_render_raises_when_no_mesh_loaded(self) -> None:
        r = MeshShaderRenderer()
        vp = np.eye(4, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            r.render(vp, width=64, height=64)

    def test_render_returns_image_shape(self) -> None:
        r = MeshShaderRenderer()
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        r.load_mesh(vertices=vertices, faces=faces)
        vp = np.eye(4, dtype=np.float32)
        img = r.render(vp, width=32, height=32)
        self.assertEqual(img.shape, (32, 32, 4))
        self.assertEqual(img.dtype, np.uint8)


class MeshShaderRendererCloseTests(unittest.TestCase):
    def test_close_clears_all_data(self) -> None:
        with patch('metal_mesh.is_mesh_shader_supported', return_value=True):
            r = MeshShaderRenderer()
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
            faces = np.array([[0, 1, 2]], dtype=np.uint32)
            r.load_mesh(vertices=vertices, faces=faces)
        r.close()
        self.assertIsNone(r._vertices)
        self.assertIsNone(r._normals)
        self.assertIsNone(r._uvs)
        self.assertIsNone(r._faces)
        self.assertIsNone(r._meshlets)
        self.assertIsNone(r._meshlet_indices)


if __name__ == "__main__":
    unittest.main()
