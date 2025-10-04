import logging
import spacy
import networkx as nx
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
import json
import pickle
from pathlib import Path

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from config import config

logger = logging.getLogger(__name__)

class MedicalKnowledgeGraph:
    """Knowledge Graph for Emergency Medicine entities and relationships"""
    
    def __init__(self, use_neo4j: bool = False):
        """
        Initialize the knowledge graph
        
        Args:
            use_neo4j: Whether to use Neo4j (requires Neo4j installation) or NetworkX
        """
        self.use_neo4j = use_neo4j and NEO4J_AVAILABLE
        
        # Initialize spaCy for NER (using scientific model)
        try:
            self.nlp = spacy.load("en_core_sci_scibert")
            logger.info("Loaded scientific spaCy model: en_core_sci_scibert")
        except OSError:
            try:
                # Fallback to standard model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded standard spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("No spaCy model found. Install scientific model with: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz")
                self.nlp = None
        
        # Medical entities patterns for emergency medicine (expanded)
        self.medical_patterns = {
            'symptoms': [
                'chest pain', 'shortness of breath', 'abdominal pain', 'headache', 'fever',
                'nausea', 'vomiting', 'dizziness', 'syncope', 'palpitations', 'dyspnea',
                'tachycardia', 'bradycardia', 'hypotension', 'hypertension', 'seizure',
                'pain', 'bleeding', 'swelling', 'rash', 'fatigue', 'weakness'
            ],
            'conditions': [
                'myocardial infarction', 'stroke', 'pneumonia', 'sepsis', 'anaphylaxis',
                'pulmonary embolism', 'aortic dissection', 'acute coronary syndrome',
                'heart failure', 'shock', 'trauma', 'fracture', 'laceration', 'burns',
                'gastroenteritis', 'epistaxis', 'endocarditis', 'brue', 'hysterotomy',
                'epilepticus', 'retinal artery occlusion', 'bronchiolitis'
            ],
            'medications': [
                'epinephrine', 'atropine', 'adenosine', 'amiodarone', 'dopamine',
                'norepinephrine', 'morphine', 'fentanyl', 'midazolam', 'lorazepam',
                'aspirin', 'nitroglycerin', 'furosemide', 'metoprolol', 'lisinopril',
                'ketamine', 'ketorolac', 'glucose', 'oxygen', 'thrombolysis'
            ],
            'procedures': [
                'intubation', 'defibrillation', 'cardioversion', 'chest tube', 'central line',
                'lumbar puncture', 'paracentesis', 'thoracentesis', 'cricothyrotomy',
                'pericardiocentesis', 'ultrasound', 'CT scan', 'MRI', 'X-ray', 'ECG',
                'packing', 'transfusion', 'screening', 'testing', 'prediction', 'tourniquet'
            ],
            'anatomy': [
                'heart', 'lung', 'brain', 'liver', 'kidney', 'spleen', 'pancreas',
                'aorta', 'ventricle', 'atrium', 'artery', 'vein', 'trachea', 'bronchus',
                'nasal', 'septal', 'chest', 'pediatric', 'neonatal', 'maternal'
            ]
        }
        
        if self.use_neo4j:
            self._init_neo4j()
        else:
            self.graph = nx.DiGraph()
            
        self.entity_cache = {}
        self.relationship_cache = defaultdict(list)
    
    def _init_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                config.neo4j_uri,
                auth=(config.neo4j_user, config.neo4j_password)
            )
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            logger.info("Falling back to NetworkX")
            self.use_neo4j = False
            self.graph = nx.DiGraph()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary of entity types and their instances
        """
        entities = defaultdict(list)
        
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        
        # Extract predefined medical entities (improved partial matching)
        for entity_type, patterns in self.medical_patterns.items():
            for pattern in patterns:
                # Use partial matching - check if pattern appears anywhere in text
                if pattern in text_lower:
                    # Find the actual occurrence in original text for proper capitalization
                    import re
                    matches = re.findall(rf'\b\w*{re.escape(pattern)}\w*\b', text, re.IGNORECASE)
                    for match in matches:
                        if len(match.strip()) > 2:  # Avoid very short matches
                            entities[entity_type].append(match.strip())
        
        # Use spaCy SciBERT for medical NER if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                # Include medical entity types from SciBERT
                medical_labels = [
                    'CHEMICAL',      # Drugs, medications
                    'DISEASE',       # Diseases, conditions
                    'ANATOMY',       # Body parts, organs
                    'PROCEDURE',     # Medical procedures
                    'SYMPTOM',       # Symptoms
                    'TREATMENT',     # Treatments
                    'DRUG',          # Alternative drug label
                    'CONDITION',     # Alternative condition label
                    'MEDICATION',    # Alternative medication label
                    'PERSON',        # People (doctors, patients)
                    'ORG',          # Organizations (hospitals)
                    'GPE'           # Locations (cities, countries)
                ]
                
                if ent.label_ in medical_labels:
                    # Normalize entity type
                    entity_type = self._normalize_entity_type(ent.label_)
                    entity_text = ent.text.strip()
                    
                    # Only add meaningful entities (length > 2, not just numbers)
                    if len(entity_text) > 2 and not entity_text.isdigit():
                        entities[entity_type].append(entity_text)
                
                # Log detected entities for debugging
                logger.debug(f"spaCy entity: '{ent.text}' -> {ent.label_}")
        
        # Remove duplicates and clean up
        cleaned_entities = {}
        for entity_type, entity_list in entities.items():
            # Remove duplicates (case-insensitive)
            unique_entities = []
            seen = set()
            for entity in entity_list:
                entity_lower = entity.lower()
                if entity_lower not in seen and len(entity.strip()) > 2:
                    unique_entities.append(entity)
                    seen.add(entity_lower)
            
            if unique_entities:  # Only add if we have entities
                cleaned_entities[entity_type] = unique_entities
        
        # Log extracted entities for debugging
        if cleaned_entities:
            logger.info(f"Extracted entities: {dict(cleaned_entities)}")
        else:
            logger.debug("No entities extracted from text chunk")
        
        return cleaned_entities
    
    def _normalize_entity_type(self, spacy_label: str) -> str:
        """Normalize spaCy entity labels to our entity types"""
        label_mapping = {
            'CHEMICAL': 'medications',
            'DRUG': 'medications', 
            'MEDICATION': 'medications',
            'DISEASE': 'conditions',
            'CONDITION': 'conditions',
            'ANATOMY': 'anatomy',
            'PROCEDURE': 'procedures',
            'TREATMENT': 'procedures',
            'SYMPTOM': 'symptoms',
            'PERSON': 'persons',
            'ORG': 'organizations',
            'GPE': 'locations'
        }
        return label_mapping.get(spacy_label, spacy_label.lower())
    
    def extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between entities
        
        Args:
            text: Input text
            entities: Extracted entities
            
        Returns:
            List of (entity1, relationship, entity2) tuples
        """
        relationships = []
        text_lower = text.lower()
        
        # Define relationship patterns for emergency medicine
        relationship_patterns = [
            # Symptom-Condition relationships
            (r'(\w+)\s+(?:causes?|leads? to|results? in)\s+(\w+)', 'CAUSES'),
            (r'(\w+)\s+(?:indicates?|suggests?|shows?)\s+(\w+)', 'INDICATES'),
            (r'(\w+)\s+(?:associated with|related to)\s+(\w+)', 'ASSOCIATED_WITH'),
            
            # Treatment relationships
            (r'(?:treat|give|administer)\s+(\w+)\s+(?:with|using)\s+(\w+)', 'TREATED_WITH'),
            (r'(\w+)\s+(?:for|to treat)\s+(\w+)', 'TREATS'),
            
            # Diagnostic relationships
            (r'(\w+)\s+(?:diagnosed by|detected by)\s+(\w+)', 'DIAGNOSED_BY'),
            (r'(\w+)\s+(?:shows?|reveals?)\s+(\w+)', 'SHOWS'),
        ]
        
        # Extract relationships based on patterns
        import re
        for pattern, rel_type in relationship_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entity1 = match.group(1)
                entity2 = match.group(2)
                relationships.append((entity1, rel_type, entity2))
        
        # Extract co-occurrence relationships
        all_entities = []
        for entity_list in entities.values():
            all_entities.extend(entity_list)
        
        # Create co-occurrence relationships for entities in the same text
        for i, entity1 in enumerate(all_entities):
            for entity2 in all_entities[i+1:]:
                if entity1 != entity2:
                    relationships.append((entity1, 'CO_OCCURS_WITH', entity2))
        
        return relationships
    
    def add_to_graph(self, entities: Dict[str, List[str]], relationships: List[Tuple[str, str, str]], chunk_id: str):
        """
        Add entities and relationships to the knowledge graph
        
        Args:
            entities: Dictionary of entities by type
            relationships: List of relationships
            chunk_id: Source chunk identifier
        """
        if self.use_neo4j:
            self._add_to_neo4j(entities, relationships, chunk_id)
        else:
            self._add_to_networkx(entities, relationships, chunk_id)
    
    def _add_to_networkx(self, entities: Dict[str, List[str]], relationships: List[Tuple[str, str, str]], chunk_id: str):
        """Add to NetworkX graph"""
        # Add entity nodes
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if not self.graph.has_node(entity):
                    self.graph.add_node(entity, type=entity_type, sources=[chunk_id])
                else:
                    # Update sources
                    sources = self.graph.nodes[entity].get('sources', [])
                    if chunk_id not in sources:
                        sources.append(chunk_id)
                        self.graph.nodes[entity]['sources'] = sources
        
        # Add relationship edges
        for entity1, rel_type, entity2 in relationships:
            if self.graph.has_node(entity1) and self.graph.has_node(entity2):
                if self.graph.has_edge(entity1, entity2):
                    # Update relationship count
                    weight = self.graph.edges[entity1, entity2].get('weight', 0) + 1
                    self.graph.edges[entity1, entity2]['weight'] = weight
                else:
                    self.graph.add_edge(entity1, entity2, relationship=rel_type, weight=1, sources=[chunk_id])
    
    def _add_to_neo4j(self, entities: Dict[str, List[str]], relationships: List[Tuple[str, str, str]], chunk_id: str):
        """Add to Neo4j database"""
        with self.driver.session() as session:
            # Add entities
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    session.run(
                        "MERGE (e:Entity {name: $name, type: $type}) "
                        "ON CREATE SET e.sources = [$source] "
                        "ON MATCH SET e.sources = e.sources + [$source]",
                        name=entity, type=entity_type, source=chunk_id
                    )
            
            # Add relationships
            for entity1, rel_type, entity2 in relationships:
                session.run(
                    "MATCH (e1:Entity {name: $entity1}), (e2:Entity {name: $entity2}) "
                    "MERGE (e1)-[r:RELATES {type: $rel_type}]->(e2) "
                    "ON CREATE SET r.weight = 1, r.sources = [$source] "
                    "ON MATCH SET r.weight = r.weight + 1, r.sources = r.sources + [$source]",
                    entity1=entity1, entity2=entity2, rel_type=rel_type, source=chunk_id
                )
    
    def query_related_entities(self, entity: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Find entities related to the given entity
        
        Args:
            entity: Entity to find relationships for
            max_depth: Maximum relationship depth to search
            
        Returns:
            List of related entities with relationship information
        """
        if self.use_neo4j:
            return self._query_neo4j_related(entity, max_depth)
        else:
            return self._query_networkx_related(entity, max_depth)
    
    def _query_networkx_related(self, entity: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Query related entities in NetworkX"""
        related = []
        
        if entity not in self.graph:
            return related
        
        # Use BFS to find related entities
        visited = set()
        queue = [(entity, 0)]
        
        while queue:
            current_entity, depth = queue.pop(0)
            
            if depth >= max_depth or current_entity in visited:
                continue
                
            visited.add(current_entity)
            
            # Get neighbors
            for neighbor in self.graph.neighbors(current_entity):
                if neighbor not in visited:
                    edge_data = self.graph.edges[current_entity, neighbor]
                    related.append({
                        'entity': neighbor,
                        'relationship': edge_data.get('relationship', 'RELATED'),
                        'weight': edge_data.get('weight', 1),
                        'depth': depth + 1,
                        'type': self.graph.nodes[neighbor].get('type', 'unknown')
                    })
                    queue.append((neighbor, depth + 1))
        
        # Sort by weight (relevance)
        return sorted(related, key=lambda x: x['weight'], reverse=True)
    
    def _query_neo4j_related(self, entity: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Query related entities in Neo4j"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (start:Entity {name: $entity})-[r:RELATES*1..$max_depth]-(related:Entity) "
                "RETURN related.name as entity, related.type as type, "
                "reduce(weight = 0, rel in r | weight + rel.weight) as total_weight, "
                "length(r) as depth",
                entity=entity, max_depth=max_depth
            )
            
            related = []
            for record in result:
                related.append({
                    'entity': record['entity'],
                    'type': record['type'],
                    'weight': record['total_weight'],
                    'depth': record['depth'],
                    'relationship': 'RELATED'
                })
            
            return sorted(related, key=lambda x: x['weight'], reverse=True)
    
    def save_graph(self, filepath: str):
        """Save the knowledge graph to disk"""
        if not self.use_neo4j:
            with open(filepath, 'wb') as f:
                pickle.dump(self.graph, f)
            logger.info(f"Saved NetworkX graph to {filepath}")
    
    def load_graph(self, filepath: str):
        """Load the knowledge graph from disk"""
        if not self.use_neo4j and Path(filepath).exists():
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
            logger.info(f"Loaded NetworkX graph from {filepath}")
    
    def clear_graph(self):
        """Clear the entire knowledge graph"""
        if self.use_neo4j:
            with self.driver.session() as session:
                session.run("MATCH (n:Entity) DETACH DELETE n")
                logger.info("Cleared Neo4j knowledge graph")
        else:
            self.graph.clear()
            logger.info("Cleared NetworkX knowledge graph")
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if self.use_neo4j:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (n:Entity) RETURN count(n) as nodes "
                    "UNION "
                    "MATCH ()-[r:RELATES]->() RETURN count(r) as relationships"
                )
                records = list(result)
                return {
                    'nodes': records[0]['nodes'] if records else 0,
                    'relationships': records[1]['relationships'] if len(records) > 1 else 0
                }
        else:
            return {
                'nodes': self.graph.number_of_nodes(),
                'relationships': self.graph.number_of_edges(),
                'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
            }
    
    def close(self):
        """Close database connections"""
        if self.use_neo4j and hasattr(self, 'driver'):
            self.driver.close()