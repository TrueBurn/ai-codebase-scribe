#!/usr/bin/env python3

"""
Tests for src/analyzers/persistence.py

Verifies that PersistenceAnalyzer can be instantiated, PersistenceLayerInfo
can be created, and has_meaningful_persistence_content() works correctly
with sample persistence info objects.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analyzers.persistence import (
    PersistenceAnalyzer,
    PersistenceLayerInfo,
    PersistenceType,
    RelationshipInfo,
    TableInfo,
    ViewInfo,
    has_meaningful_persistence_content,
)
from src.utils.config_class import ScribeConfig


@pytest.fixture
def temp_repo(tmp_path):
    """Create a minimal temporary repository directory for instantiation tests."""
    return tmp_path


@pytest.fixture
def scribe_config():
    """Provide a default ScribeConfig."""
    config = ScribeConfig()
    config.debug = False
    return config


class TestPersistenceAnalyzerInstantiation:
    """Tests for PersistenceAnalyzer construction."""

    def test_can_be_instantiated(self, temp_repo, scribe_config):
        """Test that PersistenceAnalyzer can be created with a valid path and config."""
        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        assert analyzer is not None

    def test_repo_path_is_set(self, temp_repo, scribe_config):
        """Test that repo_path is stored correctly."""
        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        assert analyzer.repo_path == temp_repo.absolute()

    def test_config_is_set(self, temp_repo, scribe_config):
        """Test that config is stored correctly."""
        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        assert analyzer.config is scribe_config

    def test_invalid_path_raises_value_error(self, scribe_config):
        """Test that an invalid repository path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid repository path"):
            PersistenceAnalyzer(
                repo_path=Path("/nonexistent/path/that/does/not/exist"),
                config=scribe_config,
            )

    def test_initial_persistence_info_is_none(self, temp_repo, scribe_config):
        """Test that persistence_info is None before analysis."""
        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        assert analyzer.persistence_info is None

    def test_has_persistence_layer_returns_false_before_analysis(
        self, temp_repo, scribe_config
    ):
        """Test that has_persistence_layer() is False before analysis."""
        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        assert analyzer.has_persistence_layer() is False


class TestPersistenceLayerInfoCreation:
    """Tests for PersistenceLayerInfo dataclass."""

    def test_minimal_creation(self):
        """Test that PersistenceLayerInfo can be created with just persistence_type."""
        info = PersistenceLayerInfo(persistence_type=PersistenceType.FLYWAY)
        assert info.persistence_type == PersistenceType.FLYWAY

    def test_default_collections_are_empty(self):
        """Test that default collection fields are empty."""
        info = PersistenceLayerInfo(persistence_type=PersistenceType.DJANGO)
        assert info.config_files == []
        assert info.migration_files == []
        assert info.model_files == []
        assert info.migration_contents == []
        assert info.tables == {}
        assert info.views == {}
        assert info.relationships == []

    def test_tables_can_be_populated(self):
        """Test that tables dict can be populated."""
        table = TableInfo(
            name="users",
            columns=[{"name": "id", "type": "INTEGER"}],
            primary_keys=["id"],
        )
        info = PersistenceLayerInfo(
            persistence_type=PersistenceType.FLYWAY,
            tables={"users": table},
        )
        assert "users" in info.tables
        assert info.tables["users"].name == "users"

    def test_schema_data_is_empty_dict_by_default(self):
        """Test that schema_data defaults to an empty dict."""
        info = PersistenceLayerInfo(persistence_type=PersistenceType.ALEMBIC)
        assert info.schema_data == {}

    def test_all_persistence_types_can_be_used(self):
        """Test that all PersistenceType enum values can be used."""
        for persistence_type in PersistenceType:
            info = PersistenceLayerInfo(persistence_type=persistence_type)
            assert info.persistence_type == persistence_type


class TestHasMeaningfulPersistenceContent:
    """Tests for has_meaningful_persistence_content()."""

    def test_returns_false_for_none(self):
        """Test that None input returns False."""
        result = has_meaningful_persistence_content(None)
        assert result is False

    def test_returns_false_for_empty_info(self):
        """Test that an empty PersistenceLayerInfo returns False."""
        info = PersistenceLayerInfo(persistence_type=PersistenceType.FLYWAY)
        result = has_meaningful_persistence_content(info)
        assert result is False

    def test_returns_false_when_migration_files_exist_but_schema_data_empty(self):
        """Test that migration files without schema_data returns False."""
        info = PersistenceLayerInfo(
            persistence_type=PersistenceType.FLYWAY,
            migration_files=["V1__Create_users.sql"],
            schema_data={"tables": [], "relationships": [], "indexes": []},
        )
        result = has_meaningful_persistence_content(info)
        assert result is False

    def test_returns_true_when_schema_data_has_tables(self):
        """Test that schema_data with tables returns True."""
        info = PersistenceLayerInfo(
            persistence_type=PersistenceType.FLYWAY,
            migration_files=["V1__Create_users.sql"],
            schema_data={
                "tables": [{"name": "users"}],
                "relationships": [],
                "indexes": [],
            },
        )
        result = has_meaningful_persistence_content(info)
        assert result is True

    def test_returns_true_when_schema_data_has_relationships(self):
        """Test that schema_data with relationships returns True."""
        info = PersistenceLayerInfo(
            persistence_type=PersistenceType.FLYWAY,
            migration_files=["V1__Create_users.sql"],
            schema_data={
                "tables": [],
                "relationships": [{"from": "orders", "to": "users"}],
                "indexes": [],
            },
        )
        result = has_meaningful_persistence_content(info)
        assert result is True

    def test_returns_true_when_schema_data_has_indexes(self):
        """Test that schema_data with indexes returns True."""
        info = PersistenceLayerInfo(
            persistence_type=PersistenceType.FLYWAY,
            migration_files=["V1__Create_users.sql"],
            schema_data={
                "tables": [],
                "relationships": [],
                "indexes": [{"name": "idx_users_email"}],
            },
        )
        result = has_meaningful_persistence_content(info)
        assert result is True

    def test_returns_true_when_migration_contents_exist_with_schema_tables(self):
        """Test that migration_contents with tables in schema_data returns True."""
        info = PersistenceLayerInfo(
            persistence_type=PersistenceType.EFCORE,
            migration_contents=[
                {
                    "file_path": "/path/to/migration.cs",
                    "file_name": "migration.cs",
                    "content": "CreateTable(...)",
                    "type": "csharp",
                }
            ],
            schema_data={
                "tables": [{"name": "products"}],
                "relationships": [],
                "indexes": [],
            },
        )
        result = has_meaningful_persistence_content(info)
        assert result is True

    def test_non_migration_system_with_tables_returns_true(self):
        """Test that Hibernate with tables in tables dict returns True."""
        table = TableInfo(name="employees")
        info = PersistenceLayerInfo(
            persistence_type=PersistenceType.HIBERNATE,
            model_files=["/path/to/Employee.java"],
            tables={"employees": table},
        )
        result = has_meaningful_persistence_content(info)
        assert result is True


class TestPersistenceAnalyzerAnalyze:
    """Tests for PersistenceAnalyzer.analyze() on an empty repository."""

    def test_returns_none_for_empty_repo(self, temp_repo, scribe_config):
        """Test that analyze() returns None when no persistence patterns are found."""
        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        result = analyzer.analyze()
        assert result is None

    def test_has_persistence_layer_false_after_empty_analysis(
        self, temp_repo, scribe_config
    ):
        """Test that has_persistence_layer() is still False after empty analysis."""
        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        analyzer.analyze()
        assert analyzer.has_persistence_layer() is False

    def test_get_summary_without_persistence(self, temp_repo, scribe_config):
        """Test that get_summary() works when no persistence layer is detected."""
        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        analyzer.analyze()
        summary = analyzer.get_summary()
        assert isinstance(summary, dict)
        assert summary["has_persistence"] is False

    def test_flyway_detection_with_migration_files(self, temp_repo, scribe_config):
        """Test that Flyway is detected when SQL migration files are present."""
        # Create the Flyway migration directory structure
        migration_dir = temp_repo / "src" / "main" / "resources" / "db" / "migration"
        migration_dir.mkdir(parents=True)

        # Create a Flyway-style migration file
        migration_file = migration_dir / "V1__Create_users_table.sql"
        migration_file.write_text(
            "CREATE TABLE users (id BIGINT PRIMARY KEY, name VARCHAR(100));"
        )

        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        result = analyzer.analyze()

        assert result is not None
        assert result.persistence_type == PersistenceType.FLYWAY
        assert len(result.migration_files) > 0

    def test_get_summary_with_flyway(self, temp_repo, scribe_config):
        """Test that get_summary() reflects Flyway detection."""
        migration_dir = temp_repo / "src" / "main" / "resources" / "db" / "migration"
        migration_dir.mkdir(parents=True)
        migration_file = migration_dir / "V1__Init.sql"
        migration_file.write_text("CREATE TABLE items (id INT PRIMARY KEY);")

        analyzer = PersistenceAnalyzer(repo_path=temp_repo, config=scribe_config)
        analyzer.analyze()
        summary = analyzer.get_summary()

        assert summary["has_persistence"] is True
        assert summary["persistence_type"] == PersistenceType.FLYWAY.value
        assert summary["migration_count"] >= 1


class TestTableInfo:
    """Tests for the TableInfo dataclass."""

    def test_minimal_creation(self):
        """Test that TableInfo can be created with just a name."""
        table = TableInfo(name="users")
        assert table.name == "users"
        assert table.columns == []
        assert table.primary_keys == []
        assert table.foreign_keys == []
        assert table.indexes == []
        assert table.constraints == []
        assert table.migration_file is None

    def test_with_columns(self):
        """Test TableInfo with column definitions."""
        columns = [
            {"name": "id", "type": "BIGINT", "nullable": False},
            {"name": "email", "type": "VARCHAR(255)", "nullable": False},
        ]
        table = TableInfo(name="users", columns=columns, primary_keys=["id"])
        assert len(table.columns) == 2
        assert table.primary_keys == ["id"]


class TestRelationshipInfo:
    """Tests for the RelationshipInfo dataclass."""

    def test_creation(self):
        """Test that RelationshipInfo can be created."""
        relationship = RelationshipInfo(
            from_table="orders",
            to_table="users",
            from_column="user_id",
            to_column="id",
            relationship_type="many-to-one",
            constraint_name="fk_orders_user_id",
        )
        assert relationship.from_table == "orders"
        assert relationship.to_table == "users"
        assert relationship.relationship_type == "many-to-one"
        assert relationship.constraint_name == "fk_orders_user_id"
