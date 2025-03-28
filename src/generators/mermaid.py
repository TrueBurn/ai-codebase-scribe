from typing import Dict, Set, List, Optional, Tuple
import networkx as nx
from pathlib import Path
import re

class MermaidGenerator:
    """
    Generates Mermaid diagram syntax for architecture visualization.
    
    This class takes a NetworkX directed graph representing code dependencies
    and generates various Mermaid diagrams for documentation purposes.
    
    Attributes:
        graph (nx.DiGraph): The dependency graph to visualize
        direction (str): The default direction for flowcharts (TB, LR, RL, BT)
        sanitize_nodes (bool): Whether to sanitize node names for Mermaid compatibility
    """
    
    def __init__(self, graph: nx.DiGraph, direction: str = "LR", sanitize_nodes: bool = True):
        """
        Initialize the MermaidGenerator with a dependency graph.
        
        Args:
            graph (nx.DiGraph): NetworkX directed graph representing code dependencies
            direction (str, optional): Default direction for flowcharts. Defaults to "LR".
                Valid values: TB (top-bottom), LR (left-right), RL (right-left), BT (bottom-top)
            sanitize_nodes (bool, optional): Whether to sanitize node names. Defaults to True.
        
        Raises:
            ValueError: If the graph is not a NetworkX DiGraph or if direction is invalid
        """
        if not isinstance(graph, nx.DiGraph):
            raise ValueError("Graph must be a NetworkX DiGraph")
        
        valid_directions = ["TB", "LR", "RL", "BT"]
        if direction not in valid_directions:
            raise ValueError(f"Direction must be one of {valid_directions}")
        
        self.graph = graph
        self.direction = direction
        self.sanitize_nodes = sanitize_nodes
    
    def _sanitize_node_name(self, node_name: str) -> str:
        """
        Sanitize node names for Mermaid compatibility.
        
        Args:
            node_name (str): The original node name
            
        Returns:
            str: Sanitized node name safe for Mermaid syntax
        """
        # Replace characters that could break Mermaid syntax
        sanitized = re.sub(r'[^\w\-\.]', '_', str(node_name))
        # Replace dots and slashes with underscores for ID compatibility
        sanitized = sanitized.replace(".", "_").replace("/", "_")
        return sanitized
    
    def generate_class_diagram(self) -> str:
        """
        Generate a class diagram showing module relationships.
        
        Returns:
            str: Mermaid class diagram syntax
            
        Example:
            ```mermaid
            classDiagram
            class module1 {
                +function1
                +function2
            }
            class module2
            module1 --> module2
            ```
        """
        try:
            content = ["```mermaid", "classDiagram"]
            
            if not self.graph.nodes():
                content.append("    %% Empty graph - no classes to display")
                content.append("```\n")
                return "\n".join(content)
            
            # Add all nodes (modules)
            for node in self.graph.nodes():
                exports = self.graph.nodes[node].get('exports', [])
                node_name = node
                if self.sanitize_nodes:
                    node_name = self._sanitize_node_name(node)
                
                if exports:
                    content.append(f"class {node_name} {{")
                    for export in exports:
                        # Sanitize export names as well
                        export_name = export
                        if self.sanitize_nodes:
                            export_name = re.sub(r'[^\w\-]', '_', str(export))
                        content.append(f"    +{export_name}")
                    content.append("}")
                else:
                    content.append(f"class {node_name}")
            
            # Add relationships
            for src, dst in self.graph.edges():
                src_name = src
                dst_name = dst
                if self.sanitize_nodes:
                    src_name = self._sanitize_node_name(src)
                    dst_name = self._sanitize_node_name(dst)
                content.append(f"{src_name} --> {dst_name}")
            
            content.append("```\n")
            return "\n".join(content)
        except Exception as e:
            return f"```\n%% Error generating class diagram: {str(e)}\n```\n"
    
    def generate_dependency_flowchart(self, custom_direction: Optional[str] = None) -> str:
        """
        Generate a flowchart showing module dependencies.
        
        Args:
            custom_direction (Optional[str]): Override the default direction.
                Valid values: TB, LR, RL, BT
                
        Returns:
            str: Mermaid flowchart syntax
            
        Example:
            ```mermaid
            flowchart LR
                module_1[module1] --> module_2[module2]
            ```
        """
        try:
            # Use custom direction if provided, otherwise use default
            direction = custom_direction if custom_direction else self.direction
            
            # Validate direction
            valid_directions = ["TB", "LR", "RL", "BT"]
            if direction not in valid_directions:
                direction = "LR"  # Default to LR if invalid
            
            content = ["```mermaid", f"flowchart {direction}"]
            
            if not self.graph.nodes():
                content.append("    %% Empty graph - no nodes to display")
                content.append("```\n")
                return "\n".join(content)
            
            # Add nodes with unique IDs
            for node in self.graph.nodes():
                # Clean node name for Mermaid compatibility
                node_id = node
                if self.sanitize_nodes:
                    node_id = self._sanitize_node_name(node)
                content.append(f"    {node_id}[\"{node}\"]")
            
            # Add edges
            for src, dst in self.graph.edges():
                # Clean node names
                src_id = src
                dst_id = dst
                if self.sanitize_nodes:
                    src_id = self._sanitize_node_name(src)
                    dst_id = self._sanitize_node_name(dst)
                content.append(f"    {src_id} --> {dst_id}")
            
            content.append("```\n")
            return "\n".join(content)
        except Exception as e:
            return f"```\n%% Error generating dependency flowchart: {str(e)}\n```\n"
    
    def generate_package_diagram(self, custom_direction: Optional[str] = None) -> str:
        """
        Generate a package diagram showing directory-level dependencies.
        
        Args:
            custom_direction (Optional[str]): Override the default direction.
                Valid values: TB, LR, RL, BT
                
        Returns:
            str: Mermaid flowchart syntax with subgraphs for packages
            
        Example:
            ```mermaid
            flowchart TB
                subgraph package_1[package1]
                    module_1[module1]
                end
                subgraph package_2[package2]
                    module_2[module2]
                end
                module_1 --> module_2
            ```
        """
        try:
            # Use custom direction if provided, otherwise use default
            direction = custom_direction if custom_direction else self.direction
            if direction not in ["TB", "LR", "RL", "BT"]:
                direction = "TB"  # Default to TB for package diagrams if invalid
            
            # Group nodes by directory
            packages: Dict[str, Set[str]] = {}
            
            if not self.graph.nodes():
                return "```mermaid\nflowchart TB\n    %% Empty graph - no packages to display\n```\n"
            
            for node in self.graph.nodes():
                try:
                    path = Path(node)
                    pkg = str(path.parent) if path.parent.name else "root"
                    if pkg not in packages:
                        packages[pkg] = set()
                    packages[pkg].add(node)
                except Exception as e:
                    # If there's an error processing a node, skip it
                    continue
            
            content = ["```mermaid", f"flowchart {direction}"]
            
            # Add subgraphs for each package
            for pkg, modules in packages.items():
                pkg_id = pkg
                if self.sanitize_nodes:
                    pkg_id = self._sanitize_node_name(pkg)
                content.append(f"    subgraph {pkg_id}[\"{pkg}\"]")
                for module in modules:
                    mod_id = module
                    if self.sanitize_nodes:
                        mod_id = self._sanitize_node_name(module)
                    module_name = Path(module).name
                    content.append(f"        {mod_id}[\"{module_name}\"]")
                content.append("    end")
            
            # Add edges
            for src, dst in self.graph.edges():
                src_id = src
                dst_id = dst
                if self.sanitize_nodes:
                    src_id = self._sanitize_node_name(src)
                    dst_id = self._sanitize_node_name(dst)
                content.append(f"    {src_id} --> {dst_id}")
            
            content.append("```\n")
            return "\n".join(content)
        except Exception as e:
            return f"```\n%% Error generating package diagram: {str(e)}\n```\n"
    
    def generate_all_diagrams(self) -> Dict[str, str]:
        """
        Generate all three diagram types at once.
        
        Returns:
            Dict[str, str]: Dictionary containing all generated diagrams with keys:
                - 'class_diagram'
                - 'dependency_flowchart'
                - 'package_diagram'
        """
        return {
            'class_diagram': self.generate_class_diagram(),
            'dependency_flowchart': self.generate_dependency_flowchart(),
            'package_diagram': self.generate_package_diagram()
        }