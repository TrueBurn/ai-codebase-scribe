# Standard library imports
import asyncio
import hashlib
import logging
import time
import traceback
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
from ..analyzers.persistence import PersistenceAnalyzer, PersistenceType
from ..clients.base_llm import BaseLLMClient
from ..utils.cache import CacheManager
from ..utils.config_class import ScribeConfig
from ..utils.markdown_validator import MarkdownValidator
from ..utils.progress_utils import (
    create_migration_analysis_progress_bar,
)

# Constants for configuration
MIN_CONTENT_LENGTH = 100  # Minimum length for valid persistence content
# No TTL for migration cache - content-based invalidation only


async def analyze_persistence_layer(
    repo_path: Path, config: ScribeConfig, llm_client: BaseLLMClient
) -> Optional[Any]:
    """
    Analyze the persistence layer of the repository (analysis only).

    This function analyzes the repository for persistence patterns and extracts
    detailed schema information without generating documentation.

    Args:
        repo_path: Path to the repository
        config: Configuration
        llm_client: LLM client for analyzing migration contents

    Returns:
        PersistenceLayerInfo object with analysis results, or None if no persistence layer detected
    """
    try:
        # Initialize persistence analyzer
        persistence_analyzer = PersistenceAnalyzer(repo_path, config)

        # Analyze the repository for persistence patterns
        persistence_info = persistence_analyzer.analyze()

        if not persistence_info:
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "No persistence layer detected in repository", emoji="search"
                )
            else:
                logging.info("No persistence layer detected in repository")
            return None

        logger = _get_logger()
        if _use_visual_logging:
            logger.info(
                f"Detected persistence type: {persistence_info.persistence_type.value}",
                emoji="database",
            )
            logger.info(
                f"Found {logger.format_number(len(persistence_info.migration_files))} migration files",
                emoji="document",
            )
        else:
            logging.info(
                f"Detected persistence type: {persistence_info.persistence_type.value}"
            )
            logging.info(
                f"Found {len(persistence_info.migration_files)} migration files"
            )

        # Analyze migration contents with LLM to extract detailed schema information
        schema_data = {}
        if (
            hasattr(persistence_info, "migration_contents")
            and persistence_info.migration_contents
        ):
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    f"Analyzing {logger.format_number(len(persistence_info.migration_contents))} migration files for schema details...",
                    emoji="analyze",
                )
            else:
                logging.info(
                    f"Analyzing {len(persistence_info.migration_contents)} migration files for schema details..."
                )

            # Debug: Log migration file info
            for i, migration in enumerate(persistence_info.migration_contents):
                content_length = len(migration.get("content", ""))
                logging.debug(
                    f"Migration {i+1}: {migration.get('file_name', 'unknown')} ({content_length} chars)"
                )

                if config.debug:
                    content_preview = migration.get("content", "")[:300]
                    logging.debug(f"Migration content preview: {content_preview}...")

                    # Log persistence-type-specific patterns found in content
                    content = migration.get("content", "")
                    if persistence_info.persistence_type.value == "efcore":
                        # Log Entity Framework Core patterns
                        create_tables = content.count("CreateTable")
                        alter_tables = content.count("AlterTable")
                        add_columns = content.count("AddColumn")
                        create_indexes = content.count("CreateIndex")
                        logging.debug(
                            f"  EF patterns: CreateTable={create_tables}, AlterTable={alter_tables}, AddColumn={add_columns}, CreateIndex={create_indexes}"
                        )
                    elif persistence_info.persistence_type.value == "flyway":
                        # Log SQL DDL patterns for Flyway
                        create_tables = content.upper().count("CREATE TABLE")
                        alter_tables = content.upper().count("ALTER TABLE")
                        create_views = content.upper().count("CREATE VIEW")
                        create_indexes = content.upper().count("CREATE INDEX")
                        insert_statements = content.upper().count("INSERT INTO")
                        logging.debug(
                            f"  SQL patterns: CREATE TABLE={create_tables}, ALTER TABLE={alter_tables}, CREATE VIEW={create_views}, CREATE INDEX={create_indexes}, INSERT INTO={insert_statements}"
                        )
                    else:
                        # Generic pattern detection for other ORMs
                        logging.debug(
                            f"  Content type: {persistence_info.persistence_type.value} migration file"
                        )

            try:
                # Perform per-file LLM analysis with granular caching
                logging.debug(
                    "⏳ Analyzing migration files individually with granular caching..."
                )
                start_time = time.time()

                cache_manager = CacheManager(config)
                persistence_type = persistence_info.persistence_type.value

                schema_data = await _analyze_migrations_per_file(
                    llm_client,
                    cache_manager,
                    persistence_info.migration_contents,
                    persistence_type,
                    config,
                    repo_path,
                )

                analysis_time = time.time() - start_time

                tables_count = len(schema_data.get("tables", [])) if schema_data else 0
                relationships_count = (
                    len(schema_data.get("relationships", [])) if schema_data else 0
                )
                logger = _get_logger()
                if _use_visual_logging:
                    logger.success(
                        f"Per-file analysis completed in {analysis_time:.1f}s: {logger.format_number(tables_count)} tables, {logger.format_number(relationships_count)} relationships",
                        emoji="complete",
                    )
                else:
                    logging.info(
                        f"✅ Per-file analysis completed in {analysis_time:.1f}s: {tables_count} tables, {relationships_count} relationships"
                    )

                # Debug: Log schema extraction results
                if config.debug and schema_data:
                    logging.debug(f"Schema data keys: {list(schema_data.keys())}")
                    if schema_data.get("tables"):
                        logging.debug(f"First table: {schema_data['tables'][0]}")
                elif config.debug:
                    logging.debug("No schema data returned from LLM")
            except Exception as e:
                logger = _get_logger()
                if _use_visual_logging:
                    logger.warning(
                        f"Failed to analyze migration contents with LLM: {e}",
                        emoji="warning",
                    )
                    logger.info(
                        "Attempting basic pattern extraction as fallback...",
                        emoji="fallback",
                    )
                else:
                    logging.warning(
                        f"Failed to analyze migration contents with LLM: {e}"
                    )
                    logging.info("Attempting basic pattern extraction as fallback...")

                # Fallback: Try to extract basic table information using simple pattern matching
                try:
                    schema_data = _extract_basic_schema_info(
                        persistence_info.migration_contents
                    )
                    logger = _get_logger()
                    if schema_data.get("tables"):
                        if _use_visual_logging:
                            logger.info(
                                f"Fallback extraction found {logger.format_number(len(schema_data['tables']))} tables",
                                emoji="success",
                            )
                        else:
                            logging.info(
                                f"Fallback extraction found {len(schema_data['tables'])} tables"
                            )
                    else:
                        if _use_visual_logging:
                            logger.warning(
                                "Fallback extraction found no tables", emoji="warning"
                            )
                        else:
                            logging.warning("Fallback extraction found no tables")
                except Exception as fallback_error:
                    logger = _get_logger()
                    if _use_visual_logging:
                        logger.warning(
                            f"Fallback extraction also failed: {fallback_error}",
                            emoji="error",
                        )
                    else:
                        logging.warning(
                            f"Fallback extraction also failed: {fallback_error}"
                        )
                    schema_data = {
                        "tables": [],
                        "relationships": [],
                        "indexes": [],
                        "views": [],
                        "procedures": [],
                        "triggers": [],
                    }
        else:
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "No migration contents available for LLM analysis", emoji="info"
                )
            else:
                logging.info("No migration contents available for LLM analysis")
            schema_data = {
                "tables": [],
                "relationships": [],
                "indexes": [],
                "views": [],
                "procedures": [],
                "triggers": [],
            }

        # Store the extracted schema data in persistence_info for later use
        persistence_info.schema_data = schema_data

        logger = _get_logger()
        if _use_visual_logging:
            logger.info(
                f"Found {logger.format_number(len(schema_data.get('tables', [])))} tables from schema analysis",
                emoji="database",
            )
        else:
            logging.info(
                f"Found {len(schema_data.get('tables', []))} tables from schema analysis"
            )

        return persistence_info

    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(f"Error in persistence layer analysis: {e}", emoji="error")
        else:
            logging.error(f"Error in persistence layer analysis: {e}")
        if config.debug:
            logging.error(traceback.format_exc())
        return None


async def generate_persistence(
    repo_path: Path,
    file_manifest: dict,
    llm_client: BaseLLMClient,
    config: ScribeConfig,
    persistence_info: Optional[Any] = None,
) -> Optional[str]:
    """
    Generate persistence layer documentation for the repository.

    This function generates comprehensive documentation about the database schema, tables,
    relationships, and migration patterns using pre-analyzed persistence data.

    Args:
        repo_path: Path to the repository
        file_manifest: Dictionary of files in the repository
        llm_client: LLM client for generating content
        config: Configuration
        persistence_info: Pre-analyzed persistence information (optional)

    Returns:
        Formatted persistence documentation as a string, or None if no persistence layer detected
    """
    try:
        # Use pre-analyzed schema data if provided, otherwise analyze now
        if persistence_info is None:
            # No pre-analyzed data, perform full analysis
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "No pre-analyzed data provided, performing full persistence analysis...",
                    emoji="analyze",
                )
            else:
                logging.info(
                    "No pre-analyzed data provided, performing full persistence analysis..."
                )

            # Initialize persistence analyzer
            persistence_analyzer = PersistenceAnalyzer(repo_path, config)

            # Analyze the repository for persistence patterns
            persistence_info = persistence_analyzer.analyze()

            if not persistence_info:
                logger = _get_logger()
                if _use_visual_logging:
                    logger.info(
                        "No persistence layer detected in repository", emoji="search"
                    )
                else:
                    logging.info("No persistence layer detected in repository")
                return None

            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    f"Detected persistence type: {persistence_info.persistence_type.value}",
                    emoji="database",
                )
                logger.info(
                    f"Processing {logger.format_number(len(persistence_info.migration_files))} migration files",
                    emoji="processing",
                )
            else:
                logging.info(
                    f"Detected persistence type: {persistence_info.persistence_type.value}"
                )
                logging.info(
                    f"Processing {len(persistence_info.migration_files)} migration files"
                )

            # Analyze migration contents with LLM to extract detailed schema information
            schema_data = {}
            if (
                hasattr(persistence_info, "migration_contents")
                and persistence_info.migration_contents
            ):
                logger = _get_logger()
                if _use_visual_logging:
                    logger.info(
                        f"Analyzing {logger.format_number(len(persistence_info.migration_contents))} migration files for schema details...",
                        emoji="analyze",
                    )
                else:
                    logging.info(
                        f"Analyzing {len(persistence_info.migration_contents)} migration files for schema details..."
                    )

                logging.debug(
                    "⏳ Analyzing migration files individually with granular caching..."
                )
                start_time = time.time()

                cache_manager = CacheManager(config)
                persistence_type = persistence_info.persistence_type.value

                schema_data = await _analyze_migrations_per_file(
                    llm_client,
                    cache_manager,
                    persistence_info.migration_contents,
                    persistence_type,
                    config,
                    repo_path,
                )

                analysis_time = time.time() - start_time
                logger = _get_logger()
                if _use_visual_logging:
                    logger.success(
                        f"Per-file analysis completed in {analysis_time:.1f}s: {logger.format_number(len(schema_data.get('tables', [])))} tables, {logger.format_number(len(schema_data.get('relationships', [])))} relationships",
                        emoji="complete",
                    )
                else:
                    logging.info(
                        f"✅ Per-file analysis completed in {analysis_time:.1f}s: {len(schema_data.get('tables', []))} tables, {len(schema_data.get('relationships', []))} relationships"
                    )

            # Store the schema data in the persistence_info object
            persistence_info.schema_data = schema_data
        else:
            # Use pre-analyzed data - extract schema_data from the persistence_info object
            if hasattr(persistence_info, "schema_data"):
                schema_data = persistence_info.schema_data
            else:
                # If persistence_info doesn't have schema_data, assume the whole object is schema_data
                schema_data = persistence_info
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    f"Using pre-analyzed persistence data: {logger.format_number(len(schema_data.get('tables', [])))} tables, {logger.format_number(len(schema_data.get('relationships', [])))} relationships",
                    emoji="database",
                )
            else:
                logging.info(
                    f"Using pre-analyzed persistence data: {len(schema_data.get('tables', []))} tables, {len(schema_data.get('relationships', []))} relationships"
                )

        # Validate we have schema data to work with
        if not schema_data or not isinstance(schema_data, dict):
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "No schema data available for documentation generation",
                    emoji="info",
                )
            else:
                logging.info("No schema data available for documentation generation")
            return None

        # Generate persistence documentation using LLM
        try:
            logger = _get_logger()
            if _use_visual_logging:
                logger.info(
                    "Calling LLM to generate persistence documentation...",
                    emoji="document",
                )
            else:
                logging.info("Calling LLM to generate persistence documentation...")
            persistence_content = await llm_client.generate_persistence_doc(
                file_manifest, persistence_info
            )

            # Log the response for debugging
            if config.debug:
                content_preview = (
                    persistence_content[:200] if persistence_content else "None"
                )
                logging.info(f"LLM response preview: {content_preview}...")

            # Validate the generated content
            if (
                persistence_content
                and len(persistence_content.strip()) >= MIN_CONTENT_LENGTH
            ):
                # Use MarkdownValidator to check for proper formatting
                try:
                    validator = MarkdownValidator(persistence_content)
                    # Check if it's valid markdown (validate returns list of issues)
                    issues = validator.validate()

                    if config.debug:
                        logging.debug(
                            f"Markdown validation returned {len(issues)} issues (type: {type(issues)})"
                        )
                        if issues:
                            logging.debug(
                                f"First issue type: {type(issues[0])}, content: {issues[0]}"
                            )

                    if not issues or all(issue.severity != "error" for issue in issues):
                        logger = _get_logger()
                        if _use_visual_logging:
                            logger.success(
                                "Successfully generated persistence documentation using LLM",
                                emoji="document",
                            )
                        else:
                            logging.info(
                                "Successfully generated persistence documentation using LLM"
                            )
                        return persistence_content
                    else:
                        # Log the specific validation issues for debugging
                        error_issues = [
                            issue for issue in issues if issue.severity == "error"
                        ]
                        logger = _get_logger()
                        if _use_visual_logging:
                            logger.warning(
                                f"Generated content failed markdown validation with {logger.format_number(len(error_issues))} error(s), using fallback",
                                emoji="warning",
                            )
                        else:
                            logging.warning(
                                f"Generated content failed markdown validation with {len(error_issues)} error(s), using fallback"
                            )
                        for issue in error_issues[:3]:  # Log first 3 errors
                            if _use_visual_logging:
                                logger.warning(
                                    f"  Line {issue.line_number}: {issue.message}",
                                    emoji="warning",
                                )
                            else:
                                logging.warning(
                                    f"  Line {issue.line_number}: {issue.message}"
                                )
                        if len(error_issues) > 3:
                            if _use_visual_logging:
                                logger.warning(
                                    f"  ... and {len(error_issues) - 3} more errors",
                                    emoji="warning",
                                )
                            else:
                                logging.warning(
                                    f"  ... and {len(error_issues) - 3} more errors"
                                )
                except Exception as e:
                    # Validation errors should not prevent returning valid content
                    # Log the error but return the content anyway
                    logger = _get_logger()
                    if _use_visual_logging:
                        logger.warning(
                            f"Markdown validation encountered error: {type(e).__name__}: {str(e)}, but content appears valid - proceeding",
                            emoji="warning",
                        )
                    else:
                        logging.warning(
                            f"Markdown validation encountered error: {type(e).__name__}: {str(e)}, but content appears valid - proceeding"
                        )
                    if config.debug:
                        logging.error("Validation error details:")
                        logging.error(
                            f"  Content length: {len(persistence_content) if persistence_content else 0}"
                        )
                        logging.error(
                            f"  Content preview: {persistence_content[:500] if persistence_content else 'None'}..."
                        )
                        logging.error(
                            f"Validation error traceback:\n{traceback.format_exc()}"
                        )
                    # Return the content despite validation errors
                    return persistence_content

        except Exception as e:
            logger = _get_logger()
            if _use_visual_logging:
                logger.error(
                    f"Error generating persistence documentation with LLM: {e}",
                    emoji="error",
                )
            else:
                logging.error(
                    f"Error generating persistence documentation with LLM: {e}"
                )
            if config.debug:
                logging.error(traceback.format_exc())

        # Fallback to basic persistence documentation
        logger = _get_logger()
        if _use_visual_logging:
            logger.info(
                "Generating basic persistence documentation...", emoji="document"
            )
        else:
            logging.info("Generating basic persistence documentation...")
        return generate_fallback_persistence(repo_path, persistence_info, config)

    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(
                f"Error in persistence documentation generation: {e}", emoji="error"
            )
        else:
            logging.error(f"Error in persistence documentation generation: {e}")
        if config.debug:
            logging.error(traceback.format_exc())
        return None


def generate_fallback_persistence(
    repo_path: Path, persistence_info: Any, config: ScribeConfig
) -> str:
    """
    Generate comprehensive persistence documentation when LLM fails.

    This function creates a structured document based on the analyzed schema data
    from LLM migration analysis, providing detailed information about all tables,
    relationships, indexes, and migration patterns.

    Args:
        repo_path: Path to the repository
        persistence_info: PersistenceLayerInfo object with schema_data from analysis
        config: Configuration

    Returns:
        Comprehensive persistence documentation as a string using analyzed schema data
    """
    project_name = repo_path.name

    # Extract schema data from persistence_info (populated by LLM analysis)
    schema_data = getattr(persistence_info, "schema_data", {})
    tables = schema_data.get("tables", [])
    relationships = schema_data.get("relationships", [])
    indexes = schema_data.get("indexes", [])
    views = schema_data.get("views", [])

    # Build the document
    content = f"# Persistence Layer Documentation: {project_name}\n\n"
    content += "_Note: This documentation was generated from analyzed schema data without LLM enhancement._\n\n"

    # Table of Contents
    content += "## Table of Contents\n\n"
    content += "- [Overview](#overview)\n"
    content += "- [Persistence Technology](#persistence-technology)\n"
    content += "- [Database Schema](#database-schema)\n"

    if tables:
        content += "- [Tables](#tables)\n"
    if views:
        content += "- [Views](#views)\n"
    if relationships:
        content += "- [Relationships](#relationships)\n"
    if indexes:
        content += "- [Indexes](#indexes)\n"
    if persistence_info.migration_files:
        content += "- [Migrations](#migrations)\n"
    if persistence_info.config_files:
        content += "- [Configuration](#configuration)\n"

    content += "\n## Overview\n\n"

    # Generate overview based on detected patterns
    if persistence_info.persistence_type != PersistenceType.UNKNOWN:
        tech_name = _get_technology_name(persistence_info.persistence_type)
        content += (
            f"This project uses **{tech_name}** as its primary persistence technology. "
        )

        if tables:
            content += f"The database schema includes {len(tables)} table(s)"
            if views:
                content += f" and {len(views)} view(s)"
            content += ". "

        if persistence_info.migration_files:
            content += f"Database migrations are managed through {len(persistence_info.migration_files)} migration file(s). "

        if relationships:
            content += f"The schema defines {len(relationships)} relationship(s) between tables."
    else:
        content += "A persistence layer was detected but the specific technology could not be determined."

    content += "\n\n## Persistence Technology\n\n"

    # Technology details
    tech_name = _get_technology_name(persistence_info.persistence_type)
    content += f"**Primary Technology:** {tech_name}\n\n"

    if persistence_info.detected_patterns:
        content += "**Detection Confidence:**\n\n"
        for pattern, score in persistence_info.detected_patterns.items():
            confidence = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            content += f"- {pattern}: {confidence} ({score:.2f})\n"
        content += "\n"

    # Database Schema section
    content += "## Database Schema\n\n"

    if tables:
        table_count = len(tables)
        view_count = len(views)
        relationship_count = len(relationships)
        index_count = len(indexes)

        content += "The database schema consists of:\n\n"
        content += f"- **Tables:** {table_count}\n"
        if view_count > 0:
            content += f"- **Views:** {view_count}\n"
        if relationship_count > 0:
            content += f"- **Relationships:** {relationship_count}\n"
        if index_count > 0:
            content += f"- **Indexes:** {index_count}\n"
        content += "\n"
    else:
        content += (
            "No table information could be extracted from the migration files.\n\n"
        )

    # Tables section - use schema_data tables list
    if tables:
        content += "## Tables\n\n"

        for table in tables:
            table_name = table.get("name", "unknown")
            content += f"### {table_name}\n\n"

            # Migration source
            migration_file = table.get("migration_file", "")
            if migration_file:
                content += f"**Source Migration:** `{migration_file}`\n\n"

            # Columns
            columns = table.get("columns", [])
            if columns:
                content += "**Columns:**\n\n"
                content += "| Column | Type | Nullable | Constraints |\n"
                content += "|--------|------|----------|-------------|\n"

                for column in columns:
                    col_name = column.get("name", "unknown")
                    col_type = column.get("type", "unknown")
                    nullable = "Yes" if column.get("nullable", True) else "No"

                    # Build constraints string
                    constraints = []
                    if column.get("primary_key", False):
                        constraints.append("PRIMARY KEY")
                    if column.get("unique", False):
                        constraints.append("UNIQUE")
                    if column.get("default_value"):
                        constraints.append(f"DEFAULT {column.get('default_value')}")
                    constraint_str = ", ".join(constraints) if constraints else "-"

                    content += (
                        f"| {col_name} | {col_type} | {nullable} | {constraint_str} |\n"
                    )
                content += "\n"

            # Primary Keys
            primary_keys = table.get("primary_keys", [])
            if primary_keys:
                content += f"**Primary Key(s):** {', '.join(primary_keys)}\n\n"

            # Foreign Keys
            foreign_keys = table.get("foreign_keys", [])
            if foreign_keys:
                content += "**Foreign Keys:**\n\n"
                for fk in foreign_keys:
                    col = fk.get("column", "unknown")
                    ref_table = fk.get("references_table", "unknown")
                    ref_col = fk.get("references_column", "unknown")
                    on_delete = fk.get("on_delete", "")
                    on_update = fk.get("on_update", "")

                    fk_str = f"- {col} → {ref_table}.{ref_col}"
                    if on_delete:
                        fk_str += f" (ON DELETE {on_delete})"
                    if on_update:
                        fk_str += f" (ON UPDATE {on_update})"
                    content += fk_str + "\n"
                content += "\n"

            # Unique Constraints
            unique_constraints = table.get("unique_constraints", [])
            if unique_constraints:
                content += "**Unique Constraints:**\n\n"
                for uc in unique_constraints:
                    if isinstance(uc, list):
                        content += f"- ({', '.join(uc)})\n"
                    else:
                        content += f"- {uc}\n"
                content += "\n"

            # Check Constraints
            check_constraints = table.get("check_constraints", [])
            if check_constraints:
                content += "**Check Constraints:**\n\n"
                for cc in check_constraints:
                    content += f"- {cc}\n"
                content += "\n"

    # Views section
    if views:
        content += "## Views\n\n"

        for view in views:
            view_name = view.get("name", "unknown")
            content += f"### {view_name}\n\n"

            depends_on = view.get("depends_on_tables", [])
            if depends_on:
                content += f"**Dependent Tables:** {', '.join(depends_on)}\n\n"

            migration_file = view.get("migration_file", "")
            if migration_file:
                content += f"**Source Migration:** `{migration_file}`\n\n"

    # Relationships section
    if relationships:
        content += "## Relationships\n\n"

        content += "| From Table | From Column | To Table | To Column | Type |\n"
        content += "|------------|-------------|----------|-----------|------|\n"

        for rel in relationships:
            from_table = rel.get("from_table", "unknown")
            from_col = rel.get("from_column", "unknown")
            to_table = rel.get("to_table", "unknown")
            to_col = rel.get("to_column", "unknown")
            rel_type = rel.get("relationship_type", "unknown")
            content += (
                f"| {from_table} | {from_col} | {to_table} | {to_col} | {rel_type} |\n"
            )
        content += "\n"

    # Indexes section
    if indexes:
        content += "## Indexes\n\n"

        content += "| Index Name | Table | Columns | Type | Unique |\n"
        content += "|------------|-------|---------|------|--------|\n"

        for idx in indexes:
            idx_name = idx.get("name", "unnamed")
            table_name = idx.get("table", "unknown")
            columns = ", ".join(idx.get("columns", []))
            idx_type = idx.get("type", "BTREE")
            unique = "Yes" if idx.get("unique", False) else "No"
            content += (
                f"| {idx_name} | {table_name} | {columns} | {idx_type} | {unique} |\n"
            )
        content += "\n"

    # Migrations section
    if persistence_info.migration_files:
        content += "## Migrations\n\n"

        content += f"This project contains {len(persistence_info.migration_files)} migration file(s):\n\n"

        # Sort migration files for better presentation
        sorted_migrations = sorted(persistence_info.migration_files)

        for migration_file in sorted_migrations:
            migration_path = Path(migration_file)
            content += f"- `{migration_path.name}`\n"
        content += "\n"

        # Migration directory structure
        if sorted_migrations:
            sample_path = Path(sorted_migrations[0])
            migration_dir = sample_path.parent.relative_to(repo_path)
            content += f"**Migration Directory:** `{migration_dir}/`\n\n"

    # Configuration section
    if persistence_info.config_files:
        content += "## Configuration\n\n"

        content += "**Configuration Files:**\n\n"
        for config_file in persistence_info.config_files:
            config_path = Path(config_file)
            content += f"- `{config_path.name}`\n"
        content += "\n"

    # Connection information
    if persistence_info.connection_info:
        content += "**Connection Information:**\n\n"
        for key, value in persistence_info.connection_info.items():
            content += f"- **{key}:** {value}\n"
        content += "\n"

    # Add footer note
    content += "\n---\n"
    content += "_This documentation was automatically generated from database migration analysis. "
    content += f"All {len(tables)} tables and {len(relationships)} relationships were extracted from migration files._\n"

    return content


def _get_technology_name(persistence_type: PersistenceType) -> str:
    """Get the display name for a persistence technology."""
    names = {
        PersistenceType.FLYWAY: "Flyway Database Migration",
        PersistenceType.EFCORE: "Entity Framework Core",
        PersistenceType.PRISMA: "Prisma ORM",
        PersistenceType.HIBERNATE: "Hibernate ORM",
        PersistenceType.DJANGO: "Django ORM",
        PersistenceType.RAILS: "Ruby on Rails ActiveRecord",
        PersistenceType.SEQUELIZE: "Sequelize ORM",
        PersistenceType.ALEMBIC: "Alembic (SQLAlchemy)",
        PersistenceType.UNKNOWN: "Unknown Persistence Technology",
    }
    return names.get(persistence_type, "Unknown")


def _extract_basic_schema_info(
    migration_contents: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Extract basic schema information using simple pattern matching as a fallback.

    Args:
        migration_contents: List of migration file info dictionaries

    Returns:
        Dictionary containing basic extracted schema information
    """
    import re

    tables = []
    indexes = []

    for migration in migration_contents:
        content = migration.get("content", "")
        file_name = migration.get("file_name", "unknown")

        # Check migration type and extract tables accordingly
        migration_type = migration.get("type", "unknown")

        logging.debug(f"Processing {migration_type} migration: {file_name}")

        if migration_type == "sql":
            # Extract SQL CREATE TABLE statements
            create_table_pattern = (
                r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`?(\w+)`?|(\w+))\s*\("
            )
            table_matches = re.findall(create_table_pattern, content, re.IGNORECASE)

            for match in table_matches:
                # Handle both quoted and unquoted table names
                table_name = match[0] if match[0] else match[1]
                if table_name:
                    # Try to extract column information from CREATE TABLE statement
                    table_section_pattern = rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:`?{re.escape(table_name)}`?|{re.escape(table_name)})\s*\((.*?)\);"
                    table_match = re.search(
                        table_section_pattern, content, re.IGNORECASE | re.DOTALL
                    )

                    columns = []
                    if table_match:
                        table_content = table_match.group(1)

                        # Extract column definitions (simplified)
                        column_lines = [
                            line.strip() for line in table_content.split(",")
                        ]
                        for line in column_lines:
                            if line and not line.upper().startswith(
                                (
                                    "PRIMARY KEY",
                                    "FOREIGN KEY",
                                    "CONSTRAINT",
                                    "INDEX",
                                    "KEY",
                                )
                            ):
                                # Simple column pattern: column_name data_type [constraints]
                                column_match = re.match(
                                    r"(?:`?(\w+)`?|(\w+))\s+(\w+(?:\(\d+(?:,\d+)?\))?)",
                                    line.strip(),
                                    re.IGNORECASE,
                                )
                                if column_match:
                                    col_name = (
                                        column_match.group(1)
                                        if column_match.group(1)
                                        else column_match.group(2)
                                    )
                                    col_type = column_match.group(3)
                                    nullable = "NOT NULL" not in line.upper()
                                    primary_key = (
                                        "PRIMARY KEY" in line.upper()
                                        or "AUTO_INCREMENT" in line.upper()
                                    )

                                    columns.append(
                                        {
                                            "name": col_name,
                                            "type": col_type,
                                            "nullable": nullable,
                                            "primary_key": primary_key,
                                        }
                                    )

                    tables.append(
                        {
                            "name": table_name,
                            "columns": columns,
                            "migration_file": file_name,
                        }
                    )

                    logging.debug(
                        f"Extracted SQL table: {table_name} with {len(columns)} columns"
                    )

            # Extract CREATE INDEX statements from SQL
            index_pattern = r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:`?(\w+)`?|(\w+))\s+ON\s+(?:`?(\w+)`?|(\w+))"
            index_matches = re.findall(index_pattern, content, re.IGNORECASE)

            for match in index_matches:
                index_name = match[0] if match[0] else match[1]
                table_name = match[2] if match[2] else match[3]

                if index_name and table_name:
                    indexes.append(
                        {
                            "name": index_name,
                            "table": table_name,
                            "migration_file": file_name,
                        }
                    )

                    logging.debug(f"Extracted SQL index: {index_name} on {table_name}")

        else:
            # Extract Entity Framework CreateTable calls for C# migrations
            create_table_pattern = (
                r'migrationBuilder\.CreateTable\(\s*name:\s*["\']([^"\']+)["\']'
            )
            table_matches = re.findall(create_table_pattern, content)

            for table_name in table_matches:
                # Try to extract basic column information
                table_section_pattern = rf'migrationBuilder\.CreateTable\(\s*name:\s*["\']({re.escape(table_name)})["\'].*?(?=migrationBuilder\.(?:CreateTable|CreateIndex|\s*\}}))'
                table_match = re.search(table_section_pattern, content, re.DOTALL)

                columns = []
                if table_match:
                    table_content = table_match.group()

                    # Extract column definitions
                    column_pattern = r"(\w+)\s*=\s*table\.Column<([^>]+)>\([^)]*nullable:\s*([^,)]+)[^)]*\)"
                    column_matches = re.findall(column_pattern, table_content)

                    for col_name, col_type, nullable in column_matches:
                        columns.append(
                            {
                                "name": col_name,
                                "type": col_type,
                                "nullable": "true" in nullable.lower(),
                                "primary_key": False,  # Would need more complex parsing
                            }
                        )

                tables.append(
                    {
                        "name": table_name,
                        "columns": columns,
                        "migration_file": file_name,
                    }
                )

                logging.debug(
                    f"Extracted EF table: {table_name} with {len(columns)} columns"
                )

            # Extract CreateIndex calls for EF migrations
            index_pattern = r'migrationBuilder\.CreateIndex\(\s*name:\s*["\']([^"\']+)["\'].*?table:\s*["\']([^"\']+)["\']'
            index_matches = re.findall(index_pattern, content)

            for index_name, table_name in index_matches:
                indexes.append(
                    {
                        "name": index_name,
                        "table": table_name,
                        "migration_file": file_name,
                    }
                )

                logging.debug(f"Extracted EF index: {index_name} on {table_name}")

    return {
        "tables": tables,
        "relationships": [],  # Would need more complex parsing
        "indexes": indexes,
        "views": [],
        "procedures": [],
        "triggers": [],
    }


def has_persistence_layer(repo_path: Path, config: ScribeConfig) -> bool:
    """
    Check if the repository has a detectable persistence layer.

    Args:
        repo_path: Path to the repository
        config: Configuration

    Returns:
        True if persistence layer detected, False otherwise
    """
    try:
        analyzer = PersistenceAnalyzer(repo_path, config)
        persistence_info = analyzer.analyze()
        return persistence_info is not None
    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(f"Error checking for persistence layer: {e}", emoji="error")
        else:
            logging.error(f"Error checking for persistence layer: {e}")
        return False


async def _analyze_migrations_per_file(
    llm_client: BaseLLMClient,
    cache_manager: CacheManager,
    migration_contents: List[Dict[str, str]],
    persistence_type: str,
    config: ScribeConfig,
    repo_path: Path,
) -> Dict[str, Any]:
    """
    Analyze migration files individually with per-file caching, then aggregate results.

    This function implements the hybrid per-file approach:
    1. Analyze each migration file individually and cache results
    2. Aggregate individual analyses to maintain cross-migration context

    Args:
        llm_client: LLM client for analysis
        cache_manager: Cache manager for individual file caching
        migration_contents: List of migration file info dictionaries
        persistence_type: Type of persistence technology
        config: Configuration object for accessing settings
        repo_path: Path to the repository root for relative path conversion

    Returns:
        Dict containing aggregated schema information
    """
    try:
        individual_analyses = []
        cache_hits = 0
        cache_misses = 0

        logging.debug(
            f"Starting per-file analysis of {len(migration_contents)} migration files"
        )

        # Create tasks for parallel processing
        async def analyze_single_migration_task(
            i: int, migration: Dict[str, str]
        ) -> Dict[str, Any]:
            """Analyze a single migration file with caching."""
            file_name = migration.get("file_name", f"migration_{i+1}")
            file_path = migration.get("file_path", "")
            content = migration.get("content", "")

            # Generate individual file hash for caching
            content_bytes = content.encode("utf-8")
            individual_hash = hashlib.sha256(content_bytes).hexdigest()

            # Convert absolute path to relative path for cache consistency
            relative_file_path = _convert_to_relative_path(file_path, repo_path)

            # Debug logging for content hashing
            logging.debug(f"🔍 [{i+1}/{len(migration_contents)}] {file_name}:")
            logging.debug(f"   📄 Content size: {len(content_bytes)} bytes")
            logging.debug(f"   🔐 Content hash: {individual_hash[:16]}...")
            logging.debug(f"   📁 Relative path: {relative_file_path}")

            # Try to get cached individual analysis
            cached_analysis = cache_manager.get_cached_migration_analysis(
                relative_file_path
            )

            if cached_analysis:
                logging.debug("   ✅ Cache HIT - Using cached analysis")
            else:
                logging.debug("   ❌ Cache MISS - Will analyze and cache")

            if cached_analysis:
                # Cache hit - use debug level logging to reduce verbosity
                table_count = len(cached_analysis.get("tables", []))
                index_count = len(cached_analysis.get("indexes", []))
                logging.debug(
                    f"Cache HIT [{i+1}/{len(migration_contents)}]: {file_name} ({table_count} tables, {index_count} indexes)"
                )

                return {"result": cached_analysis, "cache_hit": True, "error": None}

            else:
                # Cache miss - use debug level logging to reduce verbosity
                logging.debug(
                    f"Cache MISS [{i+1}/{len(migration_contents)}]: Analyzing {file_name} ({len(content)} chars)"
                )

                try:
                    analysis_start = time.time()
                    individual_analysis = await llm_client.analyze_single_migration(
                        migration
                    )
                    analysis_time = time.time() - analysis_start

                    if individual_analysis:
                        # Only cache and use analysis if it has meaningful content
                        table_count = len(individual_analysis.get("tables", []))
                        index_count = len(individual_analysis.get("indexes", []))
                        view_count = len(individual_analysis.get("views", []))
                        relationship_count = len(
                            individual_analysis.get("relationships", [])
                        )

                        # Check if the analysis has meaningful content
                        has_content = (
                            table_count > 0
                            or index_count > 0
                            or view_count > 0
                            or relationship_count > 0
                        )

                        if has_content:
                            # Cache the successful analysis using relative path
                            logging.debug(
                                f"   💾 Caching analysis result for {relative_file_path}"
                            )
                            logging.debug(
                                f"   📊 Analysis contains: {table_count} tables, {index_count} indexes, {view_count} views, {relationship_count} relationships"
                            )

                            try:
                                cache_manager.cache_migration_analysis(
                                    relative_file_path, individual_analysis
                                )
                                logging.debug("   ✅ Successfully cached analysis")
                            except Exception as cache_error:
                                logger = _get_logger()
                                if _use_visual_logging:
                                    logger.error(
                                        f"Failed to cache analysis: {cache_error}",
                                        emoji="error",
                                    )
                                else:
                                    logging.error(
                                        f"   ❌ Failed to cache analysis: {cache_error}"
                                    )

                            logging.debug(
                                f"Individual analysis [{i+1}/{len(migration_contents)}] completed in {analysis_time:.1f}s: {table_count} tables, {index_count} indexes, {view_count} views, {relationship_count} relationships"
                            )
                        else:
                            # Don't cache empty results, but still add them to analyses for fallback processing
                            logger = _get_logger()
                            if _use_visual_logging:
                                logger.warning(
                                    f"Individual analysis [{i+1}/{len(migration_contents)}] returned empty result for {file_name} - not caching",
                                    emoji="warning",
                                )
                            else:
                                logging.warning(
                                    f"⚠️ Individual analysis [{i+1}/{len(migration_contents)}] returned empty result for {file_name} - not caching"
                                )
                            logging.debug(
                                f"   📊 Empty analysis details: {table_count} tables, {index_count} indexes, {view_count} views, {relationship_count} relationships"
                            )

                        return {
                            "result": individual_analysis,
                            "cache_hit": False,
                            "error": None,
                        }
                    else:
                        logger = _get_logger()
                        if _use_visual_logging:
                            logger.warning(
                                f"Individual analysis returned None for {file_name}",
                                emoji="warning",
                            )
                        else:
                            logging.warning(
                                f"Individual analysis returned None for {file_name}"
                            )
                        return {
                            "result": None,
                            "cache_hit": False,
                            "error": "Analysis returned None",
                        }

                except Exception as e:
                    logger = _get_logger()
                    if _use_visual_logging:
                        logger.error(
                            f"Failed to analyze individual migration {file_name}: {e}",
                            emoji="error",
                        )
                    else:
                        logging.error(
                            f"Failed to analyze individual migration {file_name}: {e}"
                        )
                    return {"result": None, "cache_hit": False, "error": str(e)}

        # Create tasks for all migration files
        tasks = [
            analyze_single_migration_task(i, migration)
            for i, migration in enumerate(migration_contents)
        ]

        # Execute all tasks in parallel with connection limiting and progress bar
        # Get concurrency limit from config to prevent connection pool exhaustion
        concurrency_limit = 8  # Default fallback

        if config.llm_provider == "bedrock" and hasattr(config, "bedrock"):
            concurrency_limit = getattr(config.bedrock, "concurrency", 8)
        elif config.llm_provider == "ollama" and hasattr(config, "ollama"):
            concurrency_limit = getattr(config.ollama, "concurrency", 1)

        logging.debug(
            f"Using concurrency limit of {concurrency_limit} for migration analysis"
        )
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def limited_task(task):
            async with semaphore:
                return await task

        limited_tasks = [limited_task(task) for task in tasks]

        with create_migration_analysis_progress_bar(len(tasks)) as progress:
            parallel_start = time.time()
            task_results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            parallel_time = time.time() - parallel_start
            progress.update(len(tasks))  # Complete the progress bar

        # Process results and collect statistics
        individual_analyses = []
        cache_hits = 0
        cache_misses = 0
        errors = 0

        for i, task_result in enumerate(task_results):
            if isinstance(task_result, Exception):
                logger = _get_logger()
                if _use_visual_logging:
                    logger.error(
                        f"Task {i+1} failed with exception: {task_result}",
                        emoji="error",
                    )
                else:
                    logging.error(f"Task {i+1} failed with exception: {task_result}")
                errors += 1
                continue

            if task_result["result"] is not None:
                individual_analyses.append(task_result["result"])

            if task_result["cache_hit"]:
                cache_hits += 1
            else:
                cache_misses += 1

            if task_result["error"]:
                errors += 1

        # Log parallel processing performance
        total_files = len(migration_contents)
        cache_hit_rate = (cache_hits / total_files * 100) if total_files > 0 else 0
        logger = _get_logger()
        if _use_visual_logging:
            logger.info(
                f"Parallel processing completed in {parallel_time:.1f}s",
                emoji="complete",
            )
            logger.info(
                f"Cache performance: {cache_hits}/{total_files} hits ({cache_hit_rate:.1f}% hit rate)",
                emoji="cache",
            )
        else:
            logging.info(f"Parallel processing completed in {parallel_time:.1f}s")
            logging.info(
                f"Cache performance: {cache_hits}/{total_files} hits ({cache_hit_rate:.1f}% hit rate)"
            )
        if errors > 0:
            if _use_visual_logging:
                logger.warning(
                    f"Encountered {logger.format_number(errors)} errors during parallel processing",
                    emoji="warning",
                )
            else:
                logging.warning(
                    f"Encountered {errors} errors during parallel processing"
                )

        if not individual_analyses:
            logger = _get_logger()
            if _use_visual_logging:
                logger.warning(
                    "No individual analyses were successful", emoji="warning"
                )
            else:
                logging.warning("No individual analyses were successful")
            return _empty_schema_structure_dict()

        # Aggregate individual analyses to maintain cross-migration context
        logger = _get_logger()
        if _use_visual_logging:
            logger.info(
                f"Aggregating {logger.format_number(len(individual_analyses))} individual analyses to detect relationships...",
                emoji="analyze",
            )
        else:
            logging.info(
                f"Aggregating {len(individual_analyses)} individual analyses to detect relationships..."
            )
        aggregation_start = time.time()

        try:
            aggregated_schema = await llm_client.aggregate_migration_analyses(
                individual_analyses
            )
            aggregation_time = time.time() - aggregation_start

            # Log final results
            final_tables = len(aggregated_schema.get("tables", []))
            final_relationships = len(aggregated_schema.get("relationships", []))
            final_indexes = len(aggregated_schema.get("indexes", []))

            logger = _get_logger()
            if _use_visual_logging:
                logger.success(
                    f"Aggregation completed in {aggregation_time:.1f}s: {logger.format_number(final_tables)} tables, {logger.format_number(final_relationships)} relationships, {logger.format_number(final_indexes)} indexes",
                    emoji="complete",
                )
            else:
                logging.info(
                    f"✅ Aggregation completed in {aggregation_time:.1f}s: {final_tables} tables, {final_relationships} relationships, {final_indexes} indexes"
                )

            return aggregated_schema

        except Exception as e:
            logger = _get_logger()
            if _use_visual_logging:
                logger.error(f"Aggregation failed: {e}", emoji="error")
                logger.info("Using simple fallback aggregation", emoji="fallback")
            else:
                logging.error(f"Aggregation failed: {e}")
                logging.info("Using simple fallback aggregation")
            return _simple_fallback_aggregation(individual_analyses)

    except Exception as e:
        logger = _get_logger()
        if _use_visual_logging:
            logger.error(f"Per-file migration analysis failed: {e}", emoji="error")
        else:
            logging.error(f"Per-file migration analysis failed: {e}")
        return _empty_schema_structure_dict()


def _simple_fallback_aggregation(
    individual_analyses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Simple aggregation without LLM when aggregation fails.

    Args:
        individual_analyses: List of individual migration analysis results

    Returns:
        Aggregated schema information
    """
    # Combine all elements from individual analyses
    all_tables = []
    all_indexes = []
    all_views = []
    all_relationships = []
    all_procedures = []
    all_triggers = []

    for analysis in individual_analyses:
        all_tables.extend(analysis.get("tables", []))
        all_indexes.extend(analysis.get("indexes", []))
        all_views.extend(analysis.get("views", []))
        all_relationships.extend(analysis.get("relationships", []))
        all_procedures.extend(analysis.get("procedures", []))
        all_triggers.extend(analysis.get("triggers", []))

    # Remove duplicate tables by name (keep last occurrence)
    unique_tables = {}
    for table in all_tables:
        unique_tables[table.get("name", "unknown")] = table

    return {
        "tables": list(unique_tables.values()),
        "indexes": all_indexes,
        "views": all_views,
        "relationships": all_relationships,
        "procedures": all_procedures,
        "triggers": all_triggers,
    }


def _empty_schema_structure_dict() -> Dict[str, Any]:
    """Return empty schema structure as a standalone function."""
    return {
        "tables": [],
        "relationships": [],
        "indexes": [],
        "views": [],
        "procedures": [],
        "triggers": [],
    }


def _convert_to_relative_path(file_path: str, repo_path: Path) -> str:
    """
    Convert absolute file path to relative path for cache consistency.
    Uses the same robust logic as CacheManager._create_cache_key() for consistency.

    Args:
        file_path: Absolute file path
        repo_path: Repository root path

    Returns:
        Relative file path normalized for cross-platform consistency
    """
    try:
        # CONSISTENCY FIX: Use same robust path resolution as CacheManager
        try:
            abs_file_path = Path(file_path).resolve()
        except (OSError, RuntimeError):
            # Fallback for cases where resolve() fails in containerized environments
            abs_file_path = Path(file_path).absolute()

        try:
            abs_repo_path = Path(repo_path).resolve()
        except (OSError, RuntimeError):
            abs_repo_path = Path(repo_path).absolute()

        # Create relative path with robust error handling
        rel_path = abs_file_path.relative_to(abs_repo_path)
        # Normalize path separators for cross-platform consistency
        relative_path = str(rel_path).replace("\\", "/").replace("//", "/")

        # Remove leading slash for consistency
        if relative_path.startswith("/"):
            relative_path = relative_path[1:]

        return relative_path

    except ValueError as e:
        # If relative path conversion fails, log the issue and use the file name as fallback
        logger = _get_logger()
        if _use_visual_logging:
            logger.warning(
                f"Failed to convert {file_path} to relative path: {e}", emoji="warning"
            )
        else:
            logging.warning(f"Failed to convert {file_path} to relative path: {e}")
        return Path(file_path).name
