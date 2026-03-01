# Standard library imports
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import visual logger, fallback to regular logging
try:
    from ..utils.visual_logger import get_visual_logger

    _use_visual_logging = True
except ImportError:
    _use_visual_logging = False


def _get_logger():
    """Get appropriate logger (visual or standard)."""
    if _use_visual_logging:
        return get_visual_logger()
    return logging.getLogger(__name__)


# Local imports
from ..utils.config_class import ScribeConfig


class PersistenceType(Enum):
    """Enumeration of supported persistence layer types."""

    FLYWAY = "flyway"
    EFCORE = "efcore"
    PRISMA = "prisma"
    HIBERNATE = "hibernate"
    DJANGO = "django"
    RAILS = "rails"
    SEQUELIZE = "sequelize"
    ALEMBIC = "alembic"
    UNKNOWN = "unknown"


@dataclass
class TableInfo:
    """Information about a database table."""

    name: str
    columns: List[Dict[str, Any]] = field(default_factory=list)
    primary_keys: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    migration_file: Optional[str] = None


@dataclass
class ViewInfo:
    """Information about a database view."""

    name: str
    definition: str
    dependent_tables: List[str] = field(default_factory=list)
    migration_file: Optional[str] = None


@dataclass
class RelationshipInfo:
    """Information about table relationships."""

    from_table: str
    to_table: str
    from_column: str
    to_column: str
    relationship_type: str  # "one-to-one", "one-to-many", "many-to-many"
    constraint_name: Optional[str] = None


@dataclass
class PersistenceLayerInfo:
    """Complete information about the persistence layer."""

    persistence_type: PersistenceType
    config_files: List[str] = field(default_factory=list)
    migration_files: List[str] = field(default_factory=list)
    model_files: List[str] = field(default_factory=list)
    migration_contents: List[Dict[str, str]] = field(
        default_factory=list
    )  # New field for LLM analysis
    schema_data: Dict[str, Any] = field(
        default_factory=dict
    )  # LLM-extracted schema information
    tables: Dict[str, TableInfo] = field(default_factory=dict)
    views: Dict[str, ViewInfo] = field(default_factory=dict)
    relationships: List[RelationshipInfo] = field(default_factory=list)
    connection_info: Dict[str, str] = field(default_factory=dict)
    detected_patterns: Dict[str, float] = field(default_factory=dict)


class PersistenceAnalyzer:
    """Analyzes persistence layer patterns and database schema information.

    This class detects various ORM and migration tools, analyzes their configuration
    and migration files, and extracts database schema information including tables,
    relationships, views, and indexes.
    """

    # Pattern definitions for different persistence types
    DETECTION_PATTERNS = {
        PersistenceType.FLYWAY: {
            "directories": [
                "db/migration",
                "src/main/resources/db/migration",
                "flyway",
                "src/main/java/db/migration",
            ],
            "file_patterns": [
                r"V\d+(_\d+)*__.*\.sql$",
                r"U\d+(_\d+)*__.*\.sql$",
                r"R__.*\.sql$",
                r"V\d+_\d+_\d+__.*\.java$",
                r"V\d+__.*\.java$",
            ],
            "config_files": [
                "flyway.conf",
                "application.properties",
                "application.yml",
            ],
            "content_patterns": [
                r"-- Flyway",
                r"CREATE TABLE",
                r"ALTER TABLE",
                r"extends BaseJavaMigration",
                r"import org.flywaydb",
            ],
            "weight": 1.0,
        },
        PersistenceType.EFCORE: {
            "directories": [
                "Migrations",
                "Data/Migrations",
                "Infrastructure/Migrations",
            ],
            "file_patterns": [r"\d{14}_.*\.cs$", r".*ModelSnapshot\.cs$"],
            "config_files": ["appsettings.json", "appsettings.Development.json"],
            "content_patterns": [
                r"DbContext",
                r"DbSet<",
                r"modelBuilder",
                r"__EFMigrationsHistory",
            ],
            "weight": 1.0,
        },
        PersistenceType.PRISMA: {
            "directories": ["prisma/migrations", "prisma/schema"],
            "file_patterns": [r".*\.prisma$"],
            "config_files": ["schema.prisma", "prisma/schema.prisma"],
            "content_patterns": [r"generator client", r"datasource db", r"model "],
            "weight": 1.0,
        },
        PersistenceType.HIBERNATE: {
            "directories": ["src/main/java", "src/test/java"],
            "file_patterns": [r".*\.java$"],
            "config_files": [
                "hibernate.cfg.xml",
                "persistence.xml",
                "application.properties",
                "application.yml",
            ],
            "content_patterns": [
                r"@Entity",
                r"@Table",
                r"@Id",
                r"@GeneratedValue",
                r"@ManyToOne",
                r"@OneToMany",
                r"@JoinColumn",
            ],
            "weight": 0.6,  # Reduced weight since Java projects are common
        },
        PersistenceType.DJANGO: {
            "directories": ["migrations"],
            "file_patterns": [r"\d{4}_.*\.py$"],
            "config_files": ["models.py", "settings.py"],
            "content_patterns": [
                r"from django.db import models",
                r"class Migration",
                r"models.Model",
            ],
            "weight": 1.0,
        },
        PersistenceType.RAILS: {
            "directories": ["db/migrate", "app/models"],
            "file_patterns": [r"\d{14}_.*\.rb$"],
            "config_files": ["database.yml"],
            "content_patterns": [
                r"ActiveRecord::Migration",
                r"create_table",
                r"< ApplicationRecord",
            ],
            "weight": 1.0,
        },
        PersistenceType.SEQUELIZE: {
            "directories": ["migrations", "models", "seeders"],
            "file_patterns": [r".*-.*\.js$"],
            "config_files": ["config.json", ".sequelizerc"],
            "content_patterns": [r"queryInterface", r"Sequelize", r"sequelize.define"],
            "weight": 1.0,
        },
        PersistenceType.ALEMBIC: {
            "directories": ["alembic/versions", "alembic"],
            "file_patterns": [r".*_.*\.py$"],
            "config_files": ["alembic.ini"],
            "content_patterns": [r"def upgrade", r"def downgrade", r"op.create_table"],
            "weight": 1.0,
        },
    }

    def __init__(self, repo_path: Path, config: ScribeConfig):
        """Initialize the persistence analyzer.

        Args:
            repo_path: Path to the repository root
            config: Configuration object
        """
        self.repo_path = Path(repo_path).absolute()
        self.config = config
        self.debug = config.debug
        self.logger = logging.getLogger(__name__)

        if self.debug:
            self.logger.setLevel(logging.DEBUG)

        # Validate repository path
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise ValueError(f"Invalid repository path: {self.repo_path}")

        self.persistence_info: Optional[PersistenceLayerInfo] = None

    def analyze(self) -> Optional[PersistenceLayerInfo]:
        """Analyze the repository for persistence layer patterns.

        Returns:
            PersistenceLayerInfo if persistence layer detected, None otherwise
        """
        self.logger.info("Starting persistence layer analysis...")

        # Get all files in the repository
        all_files = self._get_all_files()

        # Detect persistence patterns
        detected_patterns = self._detect_persistence_patterns(all_files)

        if self.debug:
            self.logger.debug(f"Detected patterns: {detected_patterns}")
            self.logger.debug(f"Total files analyzed: {len(all_files)}")
            sample_files = [str(f.relative_to(self.repo_path)) for f in all_files[:10]]
            self.logger.debug(f"Sample files: {sample_files}")

        if not detected_patterns:
            self.logger.info("No persistence layer patterns detected")
            return None

        # Get the highest scoring pattern
        primary_type = max(detected_patterns.items(), key=lambda x: x[1])[0]

        self.logger.info(f"Primary persistence type detected: {primary_type.value}")

        # Initialize persistence info
        self.persistence_info = PersistenceLayerInfo(
            persistence_type=primary_type,
            detected_patterns={
                pt.value: score for pt, score in detected_patterns.items()
            },
        )

        # Analyze specific persistence type
        if primary_type == PersistenceType.FLYWAY:
            self._analyze_flyway(all_files)
        elif primary_type == PersistenceType.EFCORE:
            self._analyze_efcore(all_files)
        elif primary_type == PersistenceType.PRISMA:
            self._analyze_prisma(all_files)
        elif primary_type == PersistenceType.HIBERNATE:
            self._analyze_hibernate(all_files)
        elif primary_type == PersistenceType.DJANGO:
            self._analyze_django(all_files)
        elif primary_type == PersistenceType.RAILS:
            self._analyze_rails(all_files)
        elif primary_type == PersistenceType.SEQUELIZE:
            self._analyze_sequelize(all_files)
        elif primary_type == PersistenceType.ALEMBIC:
            self._analyze_alembic(all_files)

        self.logger.info(
            f"Analysis complete. Found {len(self.persistence_info.tables)} tables"
        )
        return self.persistence_info

    def _get_all_files(self) -> List[Path]:
        """Get all files in the repository."""
        files = []
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                files.append(file_path)
        return files

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on common ignore patterns."""
        ignore_patterns = [
            ".git/",
            "node_modules/",
            "__pycache__/",
            ".pytest_cache/",
            "bin/",
            "obj/",
            "target/",
            "build/",
            "dist/",
        ]

        relative_path = file_path.relative_to(self.repo_path)
        path_str = str(relative_path)

        return any(pattern in path_str for pattern in ignore_patterns)

    def _detect_persistence_patterns(
        self, files: List[Path]
    ) -> Dict[PersistenceType, float]:
        """Detect persistence patterns in the repository.

        Args:
            files: List of all files in the repository

        Returns:
            Dictionary mapping persistence types to confidence scores
        """
        detected = {}

        for persistence_type, patterns in self.DETECTION_PATTERNS.items():
            score = self._calculate_pattern_score(files, patterns, persistence_type)
            if self.debug:
                self.logger.debug(
                    f"{persistence_type.value} score: {score:.3f} (threshold: 0.2)"
                )
            if score > 0.2:  # Minimum threshold (lowered to be more permissive)
                detected[persistence_type] = score

        return detected

    def _calculate_pattern_score(
        self,
        files: List[Path],
        patterns: Dict,
        persistence_type: PersistenceType = None,
    ) -> float:
        """Calculate confidence score for a persistence pattern.

        Args:
            files: List of all files
            patterns: Pattern definition for a persistence type

        Returns:
            Confidence score between 0 and 1
        """
        score = 0.0
        matches = []

        # Check directory patterns (weight: 0.2 each)
        directory_matches = 0
        if "directories" in patterns:
            for dir_pattern in patterns["directories"]:
                matching_files = [f for f in files if dir_pattern in str(f.parent)]
                if matching_files:
                    directory_matches += 1
                    matches.append(
                        f"Directory pattern '{dir_pattern}': {len(matching_files)} files"
                    )
            # Award points for directory matches (max 0.4 for multiple directories)
            if directory_matches > 0:
                score += min(0.4, directory_matches * 0.2)

        # Check file patterns (weight: 0.5 each, strongest indicator)
        file_pattern_matches = 0
        if "file_patterns" in patterns:
            # Precompile regex patterns
            compiled_patterns = [
                (re.compile(file_pattern), file_pattern)
                for file_pattern in patterns["file_patterns"]
            ]

            # Check each file against all patterns
            for f in files:
                for compiled_pattern, original_pattern in compiled_patterns:
                    if compiled_pattern.search(f.name):
                        file_pattern_matches += 1
                        matches.append(f"File pattern '{original_pattern}': {f.name}")
                        break  # Avoid counting the same file for multiple patterns

            # Award significant points for file pattern matches
            if file_pattern_matches > 0:
                score += min(0.8, file_pattern_matches * 0.4)

        # Check config files (weight: 0.3 each)
        config_matches = 0
        if "config_files" in patterns:
            for config_file in patterns["config_files"]:
                matching_files = [f for f in files if config_file in str(f)]
                if matching_files:
                    config_matches += 1
                    matches.append(f"Config file '{config_file}': found")
            # Award points for config matches
            if config_matches > 0:
                score += min(0.3, config_matches * 0.15)

        # Check content patterns (weight: varies by persistence type)
        content_matches = 0
        files_with_content_matches = []
        if "content_patterns" in patterns:
            # For Hibernate, content patterns are REQUIRED (no content = no detection)
            is_hibernate = persistence_type == PersistenceType.HIBERNATE

            # Precompile content patterns
            compiled_content_patterns = [
                re.compile(pattern) for pattern in patterns["content_patterns"]
            ]

            # Check content of relevant files
            relevant_extensions = {
                ".java",
                ".cs",
                ".py",
                ".rb",
                ".js",
                ".ts",
                ".sql",
                ".prisma",
            }
            for f in files[:500]:  # Limit to first 500 files for performance
                if f.suffix.lower() in relevant_extensions:
                    try:
                        content = f.read_text(encoding="utf-8", errors="ignore")
                        for pattern in compiled_content_patterns:
                            if pattern.search(content):
                                content_matches += 1
                                files_with_content_matches.append(f.name)
                                matches.append(f"Content pattern found in: {f.name}")
                                break  # One match per file is enough
                    except Exception:
                        continue

            # For Hibernate, require actual entity files with annotations
            if is_hibernate:
                if content_matches == 0:
                    # No entity annotations found = not a Hibernate project
                    if self.debug:
                        self.logger.debug(
                            "Hibernate pattern rejected: No @Entity annotations found"
                        )
                    return 0.0
                else:
                    # Add bonus for content matches
                    score += min(0.5, content_matches * 0.1)
                    if self.debug:
                        self.logger.debug(
                            f"Found {content_matches} files with Hibernate annotations"
                        )
            else:
                # For other persistence types, content patterns are a bonus
                if content_matches > 0:
                    score += min(0.3, content_matches * 0.05)

        # Debug logging
        if self.debug and matches:
            self.logger.debug(f"Pattern matches: {matches}")

        # Apply pattern weight and cap at 1.0
        final_score = score * patterns.get("weight", 1.0)

        if self.debug:
            self.logger.debug(
                f"Raw score: {score:.3f}, Weight: {patterns.get('weight', 1.0)}, Final: {final_score:.3f}"
            )

        return min(final_score, 1.0)

    def _analyze_flyway(self, files: List[Path]) -> None:
        """Analyze Flyway migration files."""
        # Look for both SQL and Java Flyway migration files
        sql_migration_files = [
            f for f in files if re.search(r"V\d+(_\d+)*__.*\.sql$", f.name)
        ]
        java_migration_files = [
            f
            for f in files
            if re.search(r"V\d+_\d+_\d+__.*\.java$|V\d+__.*\.java$", f.name)
        ]

        migration_files = sql_migration_files + java_migration_files
        self.persistence_info.migration_files = [str(f) for f in migration_files]

        # Parse SQL migration files
        for migration_file in sql_migration_files:
            self._parse_sql_migration(migration_file)

        # Parse Java migration files
        for migration_file in java_migration_files:
            self._parse_java_migration(migration_file)

    def _analyze_efcore(self, files: List[Path]) -> None:
        """Analyze Entity Framework Core files."""
        migration_files = [f for f in files if re.search(r"\d{14}_.*\.cs$", f.name)]
        self.persistence_info.migration_files = [str(f) for f in migration_files]

        # Find DbContext files
        context_files = []
        for file_path in files:
            if file_path.suffix == ".cs":
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if re.search(r"class.*DbContext", content):
                        context_files.append(file_path)
                except Exception:
                    continue

        self.persistence_info.model_files = [str(f) for f in context_files]

        # Store migration files for LLM analysis
        for migration_file in migration_files:
            try:
                content = migration_file.read_text(encoding="utf-8", errors="ignore")

                # Check if migration has meaningful content (not just empty Up/Down methods)
                has_content = any(
                    keyword in content
                    for keyword in [
                        "CreateTable",
                        "AlterTable",
                        "AddColumn",
                        "DropColumn",
                        "CreateIndex",
                        "DropIndex",
                        "AddForeignKey",
                        "DropForeignKey",
                    ]
                )

                if has_content:
                    migration_info = {
                        "file_path": str(migration_file),
                        "file_name": migration_file.name,
                        "content": content,
                        "type": "csharp",
                    }

                    if not hasattr(self.persistence_info, "migration_contents"):
                        self.persistence_info.migration_contents = []
                    self.persistence_info.migration_contents.append(migration_info)
                    self.logger.debug(
                        f"Added EF Core migration {migration_file.name} ({len(content)} chars)"
                    )
                else:
                    self.logger.debug(
                        f"Skipped EF Core migration {migration_file.name} (no schema content)"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Error reading EF Core migration {migration_file}: {e}"
                )

    def _analyze_prisma(self, files: List[Path]) -> None:
        """Analyze Prisma schema files."""
        schema_files = [f for f in files if f.name.endswith(".prisma")]
        self.persistence_info.model_files = [str(f) for f in schema_files]

        # Store schema files for LLM analysis
        for schema_file in schema_files:
            try:
                content = schema_file.read_text(encoding="utf-8", errors="ignore")
                migration_info = {
                    "file_path": str(schema_file),
                    "file_name": schema_file.name,
                    "content": content,
                    "type": "prisma",
                }

                if not hasattr(self.persistence_info, "migration_contents"):
                    self.persistence_info.migration_contents = []
                self.persistence_info.migration_contents.append(migration_info)
            except Exception as e:
                self.logger.warning(f"Error reading Prisma schema {schema_file}: {e}")

    def _analyze_hibernate(self, files: List[Path]) -> None:
        """Analyze Hibernate entity files."""
        entity_files = []
        for file_path in files:
            if file_path.suffix == ".java":
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    if "@Entity" in content:
                        entity_files.append(file_path)
                except Exception:
                    continue

        self.persistence_info.model_files = [str(f) for f in entity_files]

        # Store entity files for LLM analysis
        for entity_file in entity_files:
            try:
                content = entity_file.read_text(encoding="utf-8", errors="ignore")
                migration_info = {
                    "file_path": str(entity_file),
                    "file_name": entity_file.name,
                    "content": content,
                    "type": "java",
                }

                if not hasattr(self.persistence_info, "migration_contents"):
                    self.persistence_info.migration_contents = []
                self.persistence_info.migration_contents.append(migration_info)
            except Exception as e:
                self.logger.warning(
                    f"Error reading Hibernate entity {entity_file}: {e}"
                )

    def _analyze_django(self, files: List[Path]) -> None:
        """Analyze Django migration and model files."""
        migration_files = [
            f for f in files if "migrations" in str(f) and f.name.endswith(".py")
        ]
        model_files = [f for f in files if f.name == "models.py"]

        self.persistence_info.migration_files = [str(f) for f in migration_files]
        self.persistence_info.model_files = [str(f) for f in model_files]

        # Store migration files for LLM analysis
        for migration_file in migration_files:
            try:
                content = migration_file.read_text(encoding="utf-8", errors="ignore")
                migration_info = {
                    "file_path": str(migration_file),
                    "file_name": migration_file.name,
                    "content": content,
                    "type": "python",
                }

                if not hasattr(self.persistence_info, "migration_contents"):
                    self.persistence_info.migration_contents = []
                self.persistence_info.migration_contents.append(migration_info)
            except Exception as e:
                self.logger.warning(
                    f"Error reading Django migration {migration_file}: {e}"
                )

    def _analyze_rails(self, files: List[Path]) -> None:
        """Analyze Rails migration files."""
        migration_files = [
            f for f in files if "db/migrate" in str(f) and f.name.endswith(".rb")
        ]
        self.persistence_info.migration_files = [str(f) for f in migration_files]

        # Store migration files for LLM analysis
        for migration_file in migration_files:
            try:
                content = migration_file.read_text(encoding="utf-8", errors="ignore")
                migration_info = {
                    "file_path": str(migration_file),
                    "file_name": migration_file.name,
                    "content": content,
                    "type": "ruby",
                }

                if not hasattr(self.persistence_info, "migration_contents"):
                    self.persistence_info.migration_contents = []
                self.persistence_info.migration_contents.append(migration_info)
            except Exception as e:
                self.logger.warning(
                    f"Error reading Rails migration {migration_file}: {e}"
                )

    def _analyze_sequelize(self, files: List[Path]) -> None:
        """Analyze Sequelize migration files."""
        migration_files = [
            f for f in files if "migrations" in str(f) and f.name.endswith(".js")
        ]
        self.persistence_info.migration_files = [str(f) for f in migration_files]

        # Store migration files for LLM analysis
        for migration_file in migration_files:
            try:
                content = migration_file.read_text(encoding="utf-8", errors="ignore")
                migration_info = {
                    "file_path": str(migration_file),
                    "file_name": migration_file.name,
                    "content": content,
                    "type": "javascript",
                }

                if not hasattr(self.persistence_info, "migration_contents"):
                    self.persistence_info.migration_contents = []
                self.persistence_info.migration_contents.append(migration_info)
            except Exception as e:
                self.logger.warning(
                    f"Error reading Sequelize migration {migration_file}: {e}"
                )

    def _analyze_alembic(self, files: List[Path]) -> None:
        """Analyze Alembic migration files."""
        migration_files = [
            f for f in files if "alembic/versions" in str(f) and f.name.endswith(".py")
        ]
        self.persistence_info.migration_files = [str(f) for f in migration_files]

        # Store migration files for LLM analysis
        for migration_file in migration_files:
            try:
                content = migration_file.read_text(encoding="utf-8", errors="ignore")
                migration_info = {
                    "file_path": str(migration_file),
                    "file_name": migration_file.name,
                    "content": content,
                    "type": "python",
                }

                if not hasattr(self.persistence_info, "migration_contents"):
                    self.persistence_info.migration_contents = []
                self.persistence_info.migration_contents.append(migration_info)
            except Exception as e:
                self.logger.warning(
                    f"Error reading Alembic migration {migration_file}: {e}"
                )

    def _parse_sql_migration(self, file_path: Path) -> None:
        """Store SQL migration file content for LLM analysis."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Store the migration file content for later LLM analysis
            # Instead of parsing with regex, we'll let the LLM analyze the content
            migration_info = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "content": content,
                "type": "sql",
            }

            # Add to list of migration contents for LLM processing
            if not hasattr(self.persistence_info, "migration_contents"):
                self.persistence_info.migration_contents = []
            self.persistence_info.migration_contents.append(migration_info)

        except Exception as e:
            self.logger.warning(f"Error reading SQL migration {file_path}: {e}")

    def _parse_java_migration(self, file_path: Path) -> None:
        """Store Java migration file content for LLM analysis."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Store the migration file content for later LLM analysis
            migration_info = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "content": content,
                "type": "java",
            }

            # Add to list of migration contents for LLM processing
            if not hasattr(self.persistence_info, "migration_contents"):
                self.persistence_info.migration_contents = []
            self.persistence_info.migration_contents.append(migration_info)

        except Exception as e:
            self.logger.warning(f"Error reading Java migration {file_path}: {e}")

    # Legacy parsing methods removed - now using LLM-based analysis
    # All migration content parsing is handled by LLM via analyze_migration_contents()

    def has_persistence_layer(self) -> bool:
        """Check if a persistence layer was detected."""
        return self.persistence_info is not None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the persistence layer analysis."""
        if not self.persistence_info:
            return {"has_persistence": False}

        return {
            "has_persistence": True,
            "persistence_type": self.persistence_info.persistence_type.value,
            "table_count": len(self.persistence_info.tables),
            "view_count": len(self.persistence_info.views),
            "migration_count": len(self.persistence_info.migration_files),
            "relationship_count": len(self.persistence_info.relationships),
            "detected_patterns": self.persistence_info.detected_patterns,
        }


def has_meaningful_persistence_content(persistence_info: PersistenceLayerInfo) -> bool:
    """Check if persistence info contains meaningful database schema content.

    Args:
        persistence_info: PersistenceLayerInfo object from analysis

    Returns:
        True if meaningful persistence content exists, False otherwise
    """
    if not persistence_info:
        return False

    # Check for actual database content
    has_tables = len(persistence_info.tables) > 0
    has_views = len(persistence_info.views) > 0
    has_migrations = len(persistence_info.migration_files) > 0
    has_model_files = len(persistence_info.model_files) > 0
    has_migration_contents = len(persistence_info.migration_contents) > 0

    # For migration-based systems, check if we have actual migration content
    if has_migrations or has_migration_contents:
        # Check if schema_data has any content from LLM analysis
        schema_data = persistence_info.schema_data
        schema_tables = len(schema_data.get("tables", [])) > 0
        schema_relationships = len(schema_data.get("relationships", [])) > 0
        schema_indexes = len(schema_data.get("indexes", [])) > 0

        return schema_tables or schema_relationships or schema_indexes

    # For entity-based systems (like Hibernate), require actual entity files
    if persistence_info.persistence_type == PersistenceType.HIBERNATE:
        return has_model_files and (has_tables or has_views)

    # For other systems, check for any meaningful content
    return has_tables or has_views or has_model_files
