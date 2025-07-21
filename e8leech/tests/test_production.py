from fastapi.testclient import TestClient
from e8leech.api.main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "E8Leech API is running."}

def test_e8_kissing_number():
    response = client.get("/e8/kissing_number")
    assert response.status_code == 200
    assert response.json()["kissing_number"] == 240

from click.testing import CliRunner
from e8leech.cli import cli

def test_leech_kissing_number():
    response = client.get("/leech/kissing_number")
    assert response.status_code == 200
    assert response.json()["kissing_number"] == 196560

def test_cli_crypto_encrypt():
    runner = CliRunner()
    result = runner.invoke(cli, ["crypto", "encrypt"])
    assert result.exit_code == 0
    assert "Encrypting with KYBER-E8..." in result.output

def test_cli_ai_train():
    runner = CliRunner()
    result = runner.invoke(cli, ["ai", "train"])
    assert result.exit_code == 0
    assert "Training E8GNN..." in result.output

from unittest.mock import MagicMock
from e8leech.cloud.quantum import QuantumCloudProvider

def test_cli_visualize_show():
    runner = CliRunner()
    result = runner.invoke(cli, ["visualize", "show"])
    assert result.exit_code == 0
    assert "Visualizing 24D space with hologram projection..." in result.output

def test_cloud_integration():
    # Mock the cloud provider
    mock_provider = MagicMock()
    mock_provider.run_circuit.return_value = "mock_results"

    # Create a QuantumCloudProvider with the mock
    cloud_provider = QuantumCloudProvider()
    cloud_provider.device = mock_provider

    # Run a dummy circuit
    results = cloud_provider.run_circuit("dummy_circuit")
    assert results == "mock_results"
