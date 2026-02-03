"""
Tests for amplification configuration and compilation logic.

Tests cover serialization, resolution, and compilation of amplification configs.
"""

import pytest
from unittest.mock import patch
from pathlib import Path
import tempfile
import yaml

from diffing.methods.amplification.amplification_config import (
    ModuleAmplification,
    LayerAmplification,
    AmplifiedAdapter,
    AmplificationConfig,
)


class MockModule:
    """Mock module for testing."""

    def __init__(self, path: str):
        self._module = type("obj", (object,), {"__path__": path})()


class MockStandardizedTransformer:
    """Mock StandardizedTransformer for testing."""

    def __init__(self, num_layers: int = 12):
        self.num_layers = num_layers
        # Mock attentions and mlps for compile tests
        self.attentions = [MockModule("base_model.transformer.h.0.attn")]
        self.mlps = [MockModule("base_model.transformer.h.0.mlp")]


class TestModuleAmplification:
    """Test suite for ModuleAmplification."""

    def test_init(self):
        """Test ModuleAmplification initialization."""
        mod_amp = ModuleAmplification(modules="attention", weight=1.5)
        assert mod_amp.modules == "attention"
        assert mod_amp.weight == 1.5

    def test_resolve_attention(self):
        """Test resolving attention module."""
        mod_amp = ModuleAmplification(modules="attention", weight=1.5)
        result = mod_amp.resolve()
        assert result == ["attention"]

    def test_resolve_mlp(self):
        """Test resolving MLP module."""
        mod_amp = ModuleAmplification(modules="mlp", weight=2.0)
        result = mod_amp.resolve()
        assert result == ["mlp"]

    def test_resolve_all(self):
        """Test resolving all modules."""
        mod_amp = ModuleAmplification(modules="all", weight=1.0)
        result = mod_amp.resolve()
        assert set(result) == {"attention", "mlp"}

    def test_resolve_invalid_module(self):
        """Test resolving invalid module raises assertion."""
        mod_amp = ModuleAmplification(modules="invalid", weight=1.0)
        with pytest.raises(AssertionError, match="Invalid module name"):
            mod_amp.resolve()

    def test_resolve_list_single(self):
        """Test resolving list with single module amplification."""
        mod_amps = [ModuleAmplification(modules="attention", weight=1.5)]
        base_model = MockStandardizedTransformer()
        result = ModuleAmplification.resolve_list(mod_amps, base_model)
        assert result == {"attention": 1.5, "mlp": 1.0}

    def test_resolve_list_multiple_same_module(self):
        """Test resolving list with multiple amplifications for same module."""
        mod_amps = [
            ModuleAmplification(modules="attention", weight=1.5),
            ModuleAmplification(modules="attention", weight=0.5),
        ]
        base_model = MockStandardizedTransformer()
        result = ModuleAmplification.resolve_list(mod_amps, base_model)
        # Last one wins (replaces, doesn't sum)
        assert result == {"attention": 0.5, "mlp": 1.0}

    def test_resolve_list_multiple_modules(self):
        """Test resolving list with multiple different modules."""
        mod_amps = [
            ModuleAmplification(modules="attention", weight=1.5),
            ModuleAmplification(modules="mlp", weight=2.0),
        ]
        base_model = MockStandardizedTransformer()
        result = ModuleAmplification.resolve_list(mod_amps, base_model)
        assert result == {"attention": 1.5, "mlp": 2.0}

    def test_resolve_list_all_modules(self):
        """Test resolving list with 'all' modules."""
        mod_amps = [ModuleAmplification(modules="all", weight=1.0)]
        base_model = MockStandardizedTransformer()
        result = ModuleAmplification.resolve_list(mod_amps, base_model)
        assert result == {"attention": 1.0, "mlp": 1.0}

    def test_resolve_list_mixed(self):
        """Test resolving list with mixed module types."""
        mod_amps = [
            ModuleAmplification(modules="all", weight=1.0),
            ModuleAmplification(modules="attention", weight=0.5),
        ]
        base_model = MockStandardizedTransformer()
        result = ModuleAmplification.resolve_list(mod_amps, base_model)
        # "all" sets both to 1.0, then attention gets replaced with 0.5
        assert result == {"attention": 0.5, "mlp": 1.0}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        mod_amp = ModuleAmplification(modules="attention", weight=1.5)
        result = mod_amp.to_dict()
        assert result == {"modules": "attention", "weight": 1.5}

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {"modules": "mlp", "weight": 2.0}
        mod_amp = ModuleAmplification.from_dict(data)
        assert mod_amp.modules == "mlp"
        assert mod_amp.weight == 2.0


class TestLayerAmplification:
    """Test suite for LayerAmplification."""

    def test_init_single_layer(self):
        """Test LayerAmplification initialization with single layer."""
        layer_amp = LayerAmplification(
            layers=5,
            module_amplifications=[
                ModuleAmplification(modules="attention", weight=1.5)
            ],
        )
        assert layer_amp.layers == 5
        assert len(layer_amp.module_amplifications) == 1

    def test_init_layer_list(self):
        """Test LayerAmplification initialization with layer list."""
        layer_amp = LayerAmplification(
            layers=[0, 1, 2],
            module_amplifications=[
                ModuleAmplification(modules="attention", weight=1.5)
            ],
        )
        assert layer_amp.layers == [0, 1, 2]

    def test_init_all_layers(self):
        """Test LayerAmplification initialization with all layers."""
        layer_amp = LayerAmplification(
            layers="all",
            module_amplifications=[
                ModuleAmplification(modules="attention", weight=1.5)
            ],
        )
        assert layer_amp.layers == "all"

    def test_resolve_single_layer(self):
        """Test resolving single layer."""
        base_model = MockStandardizedTransformer(num_layers=12)
        layer_amp = LayerAmplification(
            layers=5,
            module_amplifications=[
                ModuleAmplification(modules="attention", weight=1.5)
            ],
        )
        layers, module_resolution = layer_amp.resolve(base_model)
        assert layers == [5]
        assert module_resolution == {"attention": 1.5, "mlp": 1.0}

    def test_resolve_layer_list(self):
        """Test resolving layer list."""
        base_model = MockStandardizedTransformer(num_layers=12)
        layer_amp = LayerAmplification(
            layers=[0, 1, 2],
            module_amplifications=[ModuleAmplification(modules="mlp", weight=2.0)],
        )
        layers, module_resolution = layer_amp.resolve(base_model)
        assert layers == [0, 1, 2]
        assert module_resolution == {"attention": 1.0, "mlp": 2.0}

    def test_resolve_all_layers(self):
        """Test resolving all layers."""
        base_model = MockStandardizedTransformer(num_layers=12)
        layer_amp = LayerAmplification(
            layers="all",
            module_amplifications=[
                ModuleAmplification(modules="attention", weight=1.5)
            ],
        )
        layers, module_resolution = layer_amp.resolve(base_model)
        assert layers == list(range(12))
        assert module_resolution == {"attention": 1.5, "mlp": 1.0}

    def test_resolve_list_single_spec(self):
        """Test resolving list with single specification."""
        base_model = MockStandardizedTransformer(num_layers=3)
        specs = [
            LayerAmplification(
                layers=1,
                module_amplifications=[
                    ModuleAmplification(modules="attention", weight=1.5)
                ],
            )
        ]
        result = LayerAmplification.resolve_list(specs, base_model)
        assert len(result) == 3
        assert result[0] == {}
        assert result[1] == {"attention": 1.5, "mlp": 1.0}
        assert result[2] == {}

    def test_resolve_list_multiple_specs_same_layer(self):
        """Test resolving list with multiple specs for same layer - overwrites at module level."""
        base_model = MockStandardizedTransformer(num_layers=3)
        specs = [
            LayerAmplification(
                layers=1,
                module_amplifications=[
                    ModuleAmplification(modules="attention", weight=1.5)
                ],
            ),
            LayerAmplification(
                layers=1,
                module_amplifications=[ModuleAmplification(modules="mlp", weight=2.0)],
            ),
        ]
        result = LayerAmplification.resolve_list(specs, base_model)
        # Multiple specs for same layer overwrite at module level
        # Spec 1: {attention: 1.5, mlp: 1.0}
        # Spec 2: {attention: 1.0, mlp: 2.0}
        # Result: {attention: 1.0, mlp: 2.0} (both overwritten)
        assert result[1] == {"attention": 1.0, "mlp": 2.0}

    def test_resolve_list_multiple_specs_different_layers(self):
        """Test resolving list with specs for different layers."""
        base_model = MockStandardizedTransformer(num_layers=3)
        specs = [
            LayerAmplification(
                layers=0,
                module_amplifications=[
                    ModuleAmplification(modules="attention", weight=1.5)
                ],
            ),
            LayerAmplification(
                layers=2,
                module_amplifications=[ModuleAmplification(modules="mlp", weight=2.0)],
            ),
        ]
        result = LayerAmplification.resolve_list(specs, base_model)
        assert result[0] == {"attention": 1.5, "mlp": 1.0}
        assert result[1] == {}
        assert result[2] == {"attention": 1.0, "mlp": 2.0}

    def test_resolve_list_all_layers(self):
        """Test resolving list with all layers specification."""
        base_model = MockStandardizedTransformer(num_layers=3)
        specs = [
            LayerAmplification(
                layers="all",
                module_amplifications=[
                    ModuleAmplification(modules="attention", weight=1.5)
                ],
            )
        ]
        result = LayerAmplification.resolve_list(specs, base_model)
        assert len(result) == 3
        for layer_result in result:
            assert layer_result == {"attention": 1.5, "mlp": 1.0}

    def test_resolve_list_module_level_overwrite(self):
        """Test module-level overwrite behavior within a layer."""
        base_model = MockStandardizedTransformer(num_layers=3)
        specs = [
            LayerAmplification(
                layers=0,
                module_amplifications=[
                    ModuleAmplification(modules="attention", weight=2.0),
                    ModuleAmplification(modules="mlp", weight=1.0),
                ],
            ),
            LayerAmplification(
                layers=0,
                module_amplifications=[
                    ModuleAmplification(modules="attention", weight=3.0)
                ],
            ),
        ]
        result = LayerAmplification.resolve_list(specs, base_model)
        # Spec 1: layer 0 → {attention: 2.0, mlp: 1.0}
        # Spec 2: layer 0 → {attention: 3.0, mlp: 1.0} (default mlp)
        # Result: layer 0 → {attention: 3.0, mlp: 1.0} (mlp preserved, attention overwritten)
        assert result[0] == {"attention": 3.0, "mlp": 1.0}
        assert result[1] == {}
        assert result[2] == {}

    def test_resolve_list_overlapping_layers(self):
        """Test resolving list with overlapping layer specifications."""
        base_model = MockStandardizedTransformer(num_layers=3)
        specs = [
            LayerAmplification(
                layers=[0, 1],
                module_amplifications=[
                    ModuleAmplification(modules="attention", weight=1.5)
                ],
            ),
            LayerAmplification(
                layers=[1, 2],
                module_amplifications=[ModuleAmplification(modules="mlp", weight=2.0)],
            ),
        ]
        result = LayerAmplification.resolve_list(specs, base_model)
        assert result[0] == {"attention": 1.5, "mlp": 1.0}
        # Layer 1 has both specs - they overwrite at module level
        # Spec 1: {attention: 1.5, mlp: 1.0}
        # Spec 2: {attention: 1.0, mlp: 2.0}
        # Result: {attention: 1.0, mlp: 2.0} (both overwritten)
        assert result[1] == {"attention": 1.0, "mlp": 2.0}
        assert result[2] == {"attention": 1.0, "mlp": 2.0}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        layer_amp = LayerAmplification(
            layers=[0, 1],
            module_amplifications=[
                ModuleAmplification(modules="attention", weight=1.5),
                ModuleAmplification(modules="mlp", weight=2.0),
            ],
        )
        result = layer_amp.to_dict()
        assert result["layers"] == [0, 1]
        assert len(result["module_amplifications"]) == 2

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "layers": [0, 1],
            "module_amplifications": [
                {"modules": "attention", "weight": 1.5},
                {"modules": "mlp", "weight": 2.0},
            ],
        }
        layer_amp = LayerAmplification.from_dict(data)
        assert layer_amp.layers == [0, 1]
        assert len(layer_amp.module_amplifications) == 2


class TestAmplifiedAdapter:
    """Test suite for AmplifiedAdapter."""

    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_adapter_id(self, mock_resolve):
        """Test adapter_id resolution."""
        mock_resolve.return_value = "test/adapter/id"
        adapter = AmplifiedAdapter(
            organism_name="test_org",
            variant="test_variant",
            layer_amplifications=[],
        )
        result = adapter.adapter_id("base_model")
        mock_resolve.assert_called_once_with("test_org", "test_variant", "base_model")
        assert result == "test/adapter/id"

    def test_resolve(self):
        """Test resolving adapter amplifications."""
        base_model = MockStandardizedTransformer(num_layers=3)
        adapter = AmplifiedAdapter(
            organism_name="test_org",
            variant="test_variant",
            layer_amplifications=[
                LayerAmplification(
                    layers=1,
                    module_amplifications=[
                        ModuleAmplification(modules="attention", weight=1.5)
                    ],
                )
            ],
        )
        result = adapter.resolve(base_model)
        assert len(result) == 3
        assert result[1] == {"attention": 1.5, "mlp": 1.0}

    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_resolve_list_single_adapter(self, mock_resolve):
        """Test resolving list with single adapter."""
        mock_resolve.return_value = "test/adapter/id"
        base_model = MockStandardizedTransformer(num_layers=3)
        adapters = [
            AmplifiedAdapter(
                organism_name="test_org",
                variant="test_variant",
                layer_amplifications=[
                    LayerAmplification(
                        layers=1,
                        module_amplifications=[
                            ModuleAmplification(modules="attention", weight=1.5)
                        ],
                    )
                ],
            )
        ]
        result = AmplifiedAdapter.resolve_list(adapters, base_model, "base_model")
        assert "test/adapter/id" in result
        assert len(result["test/adapter/id"]) == 3

    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_resolve_list_multiple_adapters(self, mock_resolve):
        """Test resolving list with multiple adapters."""
        mock_resolve.side_effect = ["adapter1", "adapter2"]
        base_model = MockStandardizedTransformer(num_layers=3)
        adapters = [
            AmplifiedAdapter(
                organism_name="org1",
                variant="var1",
                layer_amplifications=[
                    LayerAmplification(
                        layers=0,
                        module_amplifications=[
                            ModuleAmplification(modules="attention", weight=1.5)
                        ],
                    )
                ],
            ),
            AmplifiedAdapter(
                organism_name="org2",
                variant="var2",
                layer_amplifications=[
                    LayerAmplification(
                        layers=2,
                        module_amplifications=[
                            ModuleAmplification(modules="mlp", weight=2.0)
                        ],
                    )
                ],
            ),
        ]
        result = AmplifiedAdapter.resolve_list(adapters, base_model, "base_model")
        assert "adapter1" in result
        assert "adapter2" in result
        assert len(result) == 2

    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_resolve_list_empty_amplifications(self, mock_resolve):
        """Test resolving list with adapter that has no amplifications."""
        mock_resolve.return_value = "test/adapter/id"
        base_model = MockStandardizedTransformer(num_layers=3)
        adapters = [
            AmplifiedAdapter(
                organism_name="test_org",
                variant="test_variant",
                layer_amplifications=[],
            )
        ]
        result = AmplifiedAdapter.resolve_list(adapters, base_model, "base_model")
        assert len(result) == 0

    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_resolve_list_same_adapter_id(self, mock_resolve):
        """Test resolving list with multiple adapters having same adapter_id."""
        mock_resolve.return_value = "same/adapter/id"
        base_model = MockStandardizedTransformer(num_layers=3)
        adapters = [
            AmplifiedAdapter(
                organism_name="org1",
                variant="var1",
                layer_amplifications=[
                    LayerAmplification(
                        layers=0,
                        module_amplifications=[
                            ModuleAmplification(modules="attention", weight=1.5)
                        ],
                    )
                ],
            ),
            AmplifiedAdapter(
                organism_name="org2",
                variant="var2",
                layer_amplifications=[
                    LayerAmplification(
                        layers=1,
                        module_amplifications=[
                            ModuleAmplification(modules="mlp", weight=2.0)
                        ],
                    )
                ],
            ),
        ]
        result = AmplifiedAdapter.resolve_list(adapters, base_model, "base_model")
        assert "same/adapter/id" in result
        layer_results = result["same/adapter/id"]
        assert layer_results[0] == {"attention": 1.5, "mlp": 1.0}
        assert layer_results[1] == {"attention": 1.0, "mlp": 2.0}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        adapter = AmplifiedAdapter(
            organism_name="test_org",
            variant="test_variant",
            layer_amplifications=[
                LayerAmplification(
                    layers=[0, 1],
                    module_amplifications=[
                        ModuleAmplification(modules="attention", weight=1.5)
                    ],
                )
            ],
        )
        result = adapter.to_dict()
        assert result["organism_name"] == "test_org"
        assert result["variant"] == "test_variant"
        assert len(result["layer_amplifications"]) == 1

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "organism_name": "test_org",
            "variant": "test_variant",
            "layer_amplifications": [
                {
                    "layers": [0, 1],
                    "module_amplifications": [{"modules": "attention", "weight": 1.5}],
                }
            ],
        }
        adapter = AmplifiedAdapter.from_dict(data)
        assert adapter.organism_name == "test_org"
        assert adapter.variant == "test_variant"
        assert len(adapter.layer_amplifications) == 1

    def test_from_dict_default_variant(self):
        """Test deserialization with default variant."""
        data = {
            "organism_name": "test_org",
            "layer_amplifications": [],
        }
        adapter = AmplifiedAdapter.from_dict(data)
        assert adapter.variant == "default"


class TestAmplificationConfig:
    """Test suite for AmplificationConfig."""

    def test_init(self):
        """Test AmplificationConfig initialization."""
        config = AmplificationConfig(
            name="test_config",
            description="Test description",
            amplified_adapters=[],
        )
        assert config.name == "test_config"
        assert config.description == "Test description"
        assert len(config.amplified_adapters) == 0

    def test_init_defaults(self):
        """Test AmplificationConfig initialization with defaults."""
        config = AmplificationConfig(name="test_config")
        assert config.description == ""
        assert len(config.amplified_adapters) == 0

    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_resolve(self, mock_resolve):
        """Test resolving amplification config."""
        mock_resolve.return_value = "test/adapter/id"
        base_model = MockStandardizedTransformer(num_layers=3)
        config = AmplificationConfig(
            name="test_config",
            amplified_adapters=[
                AmplifiedAdapter(
                    organism_name="test_org",
                    variant="test_variant",
                    layer_amplifications=[
                        LayerAmplification(
                            layers=1,
                            module_amplifications=[
                                ModuleAmplification(modules="attention", weight=1.5)
                            ],
                        )
                    ],
                )
            ],
        )
        result = config.resolve(base_model, "base_model")
        assert "test/adapter/id" in result

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = AmplificationConfig(
            name="test_config",
            description="Test description",
            amplified_adapters=[],
        )
        result = config.to_dict()
        assert result["name"] == "test_config"
        assert result["description"] == "Test description"
        assert "adapters" in result

    def test_to_dict_with_resolved_config(self):
        """Test serialization with resolved config."""
        config = AmplificationConfig(name="test_config")
        result = config.to_dict()
        assert "resolved_config" not in result

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "name": "test_config",
            "description": "Test description",
            "adapters": [
                {
                    "organism_name": "test_org",
                    "variant": "test_variant",
                    "layer_amplifications": [],
                }
            ],
        }
        config = AmplificationConfig.from_dict(data)
        assert config.name == "test_config"
        assert config.description == "Test description"
        assert len(config.amplified_adapters) == 1

    def test_from_dict_default_description(self):
        """Test deserialization with default description."""
        data = {"name": "test_config", "adapters": []}
        config = AmplificationConfig.from_dict(data)
        assert config.description == ""

    def test_save_yaml(self):
        """Test saving config to YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = AmplificationConfig(
                name="test_config",
                description="Test description",
                amplified_adapters=[],
            )
            config.save_yaml(config_path)
            assert config_path.exists()
            with open(config_path) as f:
                data = yaml.safe_load(f)
            assert data["name"] == "test_config"

    def test_save_yaml_with_resolved_config(self):
        """Test saving config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = AmplificationConfig(name="test_config")
            config.save_yaml(config_path)
            with open(config_path) as f:
                data = yaml.safe_load(f)
            assert "resolved_config" not in data

    def test_load_yaml(self):
        """Test loading config from YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            original_config = AmplificationConfig(
                name="test_config",
                description="Test description",
                amplified_adapters=[],
            )
            original_config.save_yaml(config_path)
            loaded_config = AmplificationConfig.load_yaml(config_path)
            assert loaded_config.name == original_config.name
            assert loaded_config.description == original_config.description

    def test_compile_empty_adapters(self):
        """Test compile with no adapters returns None."""
        config = AmplificationConfig(name="test_config", amplified_adapters=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            result, _, _ = config.compile(
                Path(tmpdir), "base_model", MockStandardizedTransformer()
            )
            assert result is None

    @patch("diffing.methods.amplification.amplification_config.adapter_id_to_path")
    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_compile_single_adapter(self, mock_resolve, mock_adapter_path):
        """Test compile with single adapter."""
        mock_resolve.return_value = "test/adapter/id"
        mock_adapter_path.return_value = Path("/fake/adapter/path")

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").touch()
            (adapter_dir / "adapter_model.bin").touch()

            mock_adapter_path.return_value = adapter_dir

            config = AmplificationConfig(
                name="test_config",
                amplified_adapters=[
                    AmplifiedAdapter(
                        organism_name="test_org",
                        variant="test_variant",
                        layer_amplifications=[
                            LayerAmplification(
                                layers=1,
                                module_amplifications=[
                                    ModuleAmplification(modules="attention", weight=1.5)
                                ],
                            )
                        ],
                    )
                ],
            )

            base_dir = Path(tmpdir) / "output"
            result, config_hash, _ = config.compile(
                base_dir, "base_model", MockStandardizedTransformer()
            )

            assert result is not None
            assert config_hash is not None
            assert result == base_dir / "test_config" / "base_model" / config_hash
            assert result.exists()
            assert (result / "amplification_config.yaml").exists()

    @patch("diffing.methods.amplification.amplification_config.adapter_id_to_path")
    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_compile_multiple_adapters(self, mock_resolve, mock_adapter_path):
        """Test compile with multiple adapters."""
        # adapter_id is called twice per adapter (once in resolve_list, once in compile)
        mock_resolve.side_effect = ["adapter1", "adapter2", "adapter1", "adapter2"]

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter1_dir = Path(tmpdir) / "adapter1"
            adapter1_dir.mkdir()
            (adapter1_dir / "file1.json").touch()

            adapter2_dir = Path(tmpdir) / "adapter2"
            adapter2_dir.mkdir()
            (adapter2_dir / "file2.json").touch()

            def adapter_path_side_effect(adapter_id):
                if adapter_id == "adapter1":
                    return adapter1_dir
                return adapter2_dir

            mock_adapter_path.side_effect = adapter_path_side_effect

            config = AmplificationConfig(
                name="test_config",
                amplified_adapters=[
                    AmplifiedAdapter(
                        organism_name="org1",
                        variant="var1",
                        layer_amplifications=[
                            LayerAmplification(
                                layers=0,
                                module_amplifications=[
                                    ModuleAmplification(modules="attention", weight=1.5)
                                ],
                            )
                        ],
                    ),
                    AmplifiedAdapter(
                        organism_name="org2",
                        variant="var2",
                        layer_amplifications=[
                            LayerAmplification(
                                layers=1,
                                module_amplifications=[
                                    ModuleAmplification(modules="mlp", weight=2.0)
                                ],
                            )
                        ],
                    ),
                ],
            )

            base_dir = Path(tmpdir) / "output"
            result, config_hash, _ = config.compile(
                base_dir, "base_model", MockStandardizedTransformer()
            )

            assert result is not None
            assert config_hash is not None
            assert result == base_dir / "test_config" / "base_model" / config_hash
            assert (result / "file1.json").exists()
            assert (result / "adapter2" / "file2.json").exists()

    @patch("diffing.methods.amplification.amplification_config.adapter_id_to_path")
    @patch("diffing.methods.amplification.amplification_config.resolve_adapter_id")
    def test_compile_overwrites_existing(self, mock_resolve, mock_adapter_path):
        """Test compile overwrites existing directory."""
        mock_resolve.return_value = "test/adapter/id"

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "file.json").touch()
            mock_adapter_path.return_value = adapter_dir

            config = AmplificationConfig(
                name="test_config",
                amplified_adapters=[
                    AmplifiedAdapter(
                        organism_name="test_org",
                        variant="test_variant",
                        layer_amplifications=[
                            LayerAmplification(
                                layers=0,
                                module_amplifications=[
                                    ModuleAmplification(modules="attention", weight=1.5)
                                ],
                            )
                        ],
                    )
                ],
            )

            base_dir = Path(tmpdir) / "output"
            output_dir = base_dir / "test_config"
            output_dir.mkdir(parents=True)
            (output_dir / "old_file.txt").touch()

            result, _, _ = config.compile(
                base_dir, "base_model", MockStandardizedTransformer()
            )

            assert result is not None
            assert not (result / "old_file.txt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
