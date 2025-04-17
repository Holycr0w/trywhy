import os
import json
import re
import numpy as np
import faiss
import markdown
from datetime import datetime
import tempfile
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import PyPDF2
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import torch

# Load configuration
def load_config():
    """Load configuration from config.json or create default if not exists"""
    config_path = "config.json"
    
    if not os.path.exists(config_path):
        default_config = {
            "company_info": {
                "name": "Your Company Name",
                "logo_path": "",
                "default_styles": {
                    "primary_color": "#003366",
                    "secondary_color": "#669933",
                    "font_family": "Arial"
                }
            },
            "api_keys": {
                "openai_key": os.environ.get("OPENAI_API_KEY", "")
            },
            "knowledge_base": {
                "directory": "markdown_responses",
                "embedding_model": "all-MiniLM-L6-v2",
                "metadata_fields": ["client_industry", "proposal_success", "project_size", "key_differentiators"]
            },
            "proposal_settings": {
                "default_sections": [],
                "max_tokens_per_section": 2000
            },
            "internal_capabilities": {
                "technical": ["Cloud solutions", "AI implementation", "Data analytics"],
                "functional": ["Project management", "24/7 support", "Custom development"]
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        return default_config
    
    with open(config_path, 'r') as f:
        return json.load(f)

# Document processing functions
def extract_text_from_docx(file_path):
    """Extract text from DOCX files including tables and headers"""
    doc = Document(file_path)
    full_text = []
    
    for table in doc.tables:
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                if cell.text.strip():
                    row_text.append(cell.text.strip())
            if row_text:
                full_text.append(" | ".join(row_text))
    
    for para in doc.paragraphs:
        if para.text.strip():
            if para.style.name.startswith('Heading'):
                heading_level = int(para.style.name[-1]) if para.style.name[-1].isdigit() else 1
                prefix = '#' * heading_level + ' '
                full_text.append(f"{prefix}{para.text}")
            else:
                full_text.append(para.text)
    
    return '\n'.join(full_text)

def extract_text_from_pdf(file_path):
    """Extract text from PDF documents"""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
    return '\n'.join(text)

def extract_sections_from_rfp(rfp_text):
    """Extract structured sections from the RFP text with improved pattern matching"""
    section_patterns = [
        r'^(?:\d+\.)?(?:\d+\.)?(?:\d+\.)?\s*([A-Z][A-Za-z\s]+)$',
        r'^([A-Z][A-Z\s]+)(?:\:|\.)?\s*$',
        r'^(?:Section|SECTION)\s+\d+\s*[\:\-\.]\s*([A-Za-z\s]+)$'
    ]
    
    sections = {}
    current_section = "Overview"
    current_content = []
    
    for line in rfp_text.split('\n'):
        matched = False
        for pattern in section_patterns:
            match = re.match(pattern, line.strip())
            if match:
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                    current_content = []
                
                current_section = match.group(1).strip()
                matched = True
                break
        
        if not matched:
            current_content.append(line)
    
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def process_rfp(file_path):
    """Extract text from uploaded RFP document"""
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.md') or file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    else:
        raise ValueError("Unsupported file format. Please use DOCX, PDF, TXT or MD file.")

def expand_query(query: str) -> str:
    """Expand query with relevant synonyms and domain-specific terms"""
    domain_specific_terms = {
        "proposal": ["offer", "bid", "solution"],
        "requirements": ["needs", "specifications", "criteria"],
        "implementation": ["deployment", "execution", "rollout"],
        "support": ["maintenance", "service", "assistance"]
    }
    
    words = query.split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        for key, values in domain_specific_terms.items():
            if word.lower() == key:
                expanded_words.extend(values)
            elif word.lower() in values:
                expanded_words.append(key)
    
    return ' '.join(expanded_words)

class HierarchicalEmbeddingModel:
    """Model for hierarchical embeddings (document and section level)"""
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        
    def encode(self, texts: List[str], level: str = 'section') -> np.ndarray:
        """Generate embeddings with different pooling strategies based on level"""
        if level == 'document':
            embeddings = self.model.encode(texts, convert_to_tensor=True)
            # Use weighted pooling for document-level embeddings
            weights = np.linspace(0.1, 1.0, len(embeddings))
            weighted_embeddings = embeddings * weights[:, np.newaxis]
            return np.mean(weighted_embeddings, axis=0)
        else:
            return self.model.encode(texts)

class ProposalKnowledgeBase:
    def __init__(self, kb_directory="markdown_responses", embedding_model="all-MiniLM-L6-v2", device="cpu"):
        self.kb_directory = kb_directory
        self.model = HierarchicalEmbeddingModel(embedding_model, device=device)
        self.documents = []
        self.section_map = {}
        self.index = None
        self.metadata = []
        self.tfidf_vectorizer = TfidfVectorizer()
        
        if not os.path.exists(kb_directory):
            os.makedirs(kb_directory)
            
        self.load_documents()
    
    def load_documents(self):
        """Load all documents from the knowledge base directory"""
        self.documents = []
        self.section_map = {}
        self.metadata = []
        
        if not os.path.exists(self.kb_directory):
            return
            
        for filename in os.listdir(self.kb_directory):
            if filename.endswith('.md') or filename.endswith('.txt'):
                file_path = os.path.join(self.kb_directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                sections = self._split_into_sections(content)
                
                for section_name, section_content in sections.items():
                    doc_id = len(self.documents)
                    metadata = {
                        "client_industry": "general",
                        "proposal_success": True,
                        "project_size": "medium",
                        "key_differentiators": ["quality", "experience"]
                    }
                    
                    # Try to extract metadata from filename or content
                    if "_success_" in filename:
                        metadata["proposal_success"] = filename.split("_success_")[1].split("_")[0] == "True"
                    if "_industry_" in filename:
                        metadata["client_industry"] = filename.split("_industry_")[1].split("_")[0]
                    if "_size_" in filename:
                        metadata["project_size"] = filename.split("_size_")[1].split("_")[0]
                    
                    self.documents.append({
                        "id": doc_id,
                        "filename": filename,
                        "section_name": section_name,
                        "content": section_content,
                        "metadata": metadata
                    })
                    
                    if section_name not in self.section_map:
                        self.section_map[section_name] = []
                    self.section_map[section_name].append(doc_id)
                    self.metadata.append(metadata)
        
        print(f"Loaded {len(self.documents)} sections from {len(self.section_map)} unique section types")
        
        self._build_index()
    
    def _split_into_sections(self, content):
        """Split a document into sections based on headers"""
        sections = {}
        current_section = "Introduction"
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('# '):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                    current_content = []
                
                current_section = line[2:].strip()
            elif line.startswith('## '):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                    current_content = []
                
                current_section = line[3:].strip()
            else:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
            
        return sections
    
    def _build_index(self):
        """Build a FAISS index for fast similarity search"""
        if not self.documents:
            return
            
        texts = [doc["content"] for doc in self.documents]
        print(f"Creating embeddings for {len(texts)} documents")
        embeddings = self.model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Build TF-IDF index for sparse retrieval
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        print("Index built successfully")
    
    def hybrid_search(self, query, k=5):
        """Hybrid search combining dense and sparse retrieval"""
        if not self.index or not self.documents:
            return []
        
        # Dense retrieval
        query_embedding = self.model.encode([query])
        dense_scores, dense_indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        # Sparse retrieval (TF-IDF)
        query_tfidf = self.tfidf_vectorizer.transform([query])
        sparse_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        sparse_indices = np.argsort(-sparse_scores)[:k]
        
        # Combine results
        combined_scores = []
        combined_indices = []
        
        dense_set = set(dense_indices[0])
        sparse_set = set(sparse_indices)
        
        for idx in dense_indices[0]:
            if idx < len(self.documents) and idx >= 0:
                combined_scores.append((dense_scores[0][list(dense_indices[0]).index(idx)], idx))
        
        for idx in sparse_indices:
            if idx < len(self.documents) and idx not in dense_set:
                combined_scores.append((sparse_scores[idx], idx))
        
        # Sort by score
        combined_scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in combined_scores[:k]:
            results.append({
                "score": float(score),
                "document": self.documents[idx]
            })
        
        return results
    
    def multi_hop_search(self, initial_query, k=5):
        """Multi-hop retrieval process"""
        # First pass: broad search
        first_results = self.hybrid_search(initial_query, k=3*k)
        
        # Analyze first results to refine query
        refined_query = initial_query
        for result in first_results[:3]:
            refined_query += " " + result["document"]["content"][:200]  # Add top content snippets
        
        # Second pass: refined search
        second_results = self.hybrid_search(refined_query, k=k)
        
        # Combine and deduplicate results
        all_results = {}
        for result in first_results + second_results:
            doc_id = result["document"]["id"]
            if doc_id not in all_results:
                all_results[doc_id] = result
            else:
                # Update score if higher
                if result["score"] > all_results[doc_id]["score"]:
                    all_results[doc_id]["score"] = result["score"]
        
        # Return top k results
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)[:k]
        return sorted_results
    
    def get_section_documents(self, section_name):
        """Get all documents for a specific section name"""
        if section_name in self.section_map:
            return [self.documents[idx] for idx in self.section_map[section_name]]
        return []
    
    def get_all_section_names(self):
        """Get all unique section names in the knowledge base"""
        return list(self.section_map.keys())

class SpecialistRAGDrafter:
    def __init__(self, openai_key=None):
        self.client = OpenAI(api_key=openai_key or os.environ.get("OPENAI_API_KEY"))
    
    def generate_draft(self, section_name, rfp_section_content, relevant_kb_content, client_name):
        """Generate a draft for a specific section using retrieved documents"""
        kb_content_list = []
        for item in relevant_kb_content:
            score = item["score"]
            doc = item["document"]
            rel_indicator = "Very Relevant" if score > 0.7 else "Relevant" if score > 0.4 else "Somewhat Relevant"
            
            # Replace names with current client name
            content = doc["content"].replace("CLIENT_NAME", client_name)
            content = content.replace("COMPANY_NAME", "Your Company Name")
            
            kb_content_list.append(f"--- {rel_indicator} PAST PROPOSAL SECTION ---\nFrom: {doc['filename']}\nSection: {doc['section_name']}\n\n{content}\n")
        
        kb_content = "\n\n".join(kb_content_list)
        
        prompt = f"""
        # DRAFT GENERATION FOR {section_name}

        ## SECTION CONTENT TO ADDRESS
        {rfp_section_content}

        ## RELEVANT KNOWLEDGE BASE CONTENT
        {kb_content}

        Generate a draft response for the section "{section_name}" based on the provided content and requirements. Focus on addressing the specific requirements and leverage the knowledge base content for relevant details.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating draft for {section_name}: {str(e)}")
            return f"Error generating draft for {section_name}: {str(e)}"

class EnhancedProposalGenerator:
    def __init__(self, knowledge_base, openai_key=None):
        self.kb = knowledge_base
        self.client = OpenAI(api_key=openai_key or os.environ.get("OPENAI_API_KEY"))
        self.rfp_text = None  # Store RFP text for regeneration
        self.drafter = SpecialistRAGDrafter(openai_key)  # Specialist drafter
    
    def analyze_rfp(self, rfp_text):
        """Comprehensive RFP analysis using the new prompt"""
        self.rfp_text = rfp_text  # Store RFP text for later use
        
        prompt = f"""
        You are an expert proposal analyst. Your task is to analyze the following Request for Proposal (RFP) text and extract key information.
        I need a comprehensive, structured analysis of the following Request for Proposal (RFP). Please organize your analysis into the following specific categories with clear headings:

        1. KEY REQUIREMENTS: Extract specific functional and technical requirements that must be addressed, using exact language from the RFP where possible.

        2. DELIVERABLES: List all concrete deliverables explicitly requested in the RFP.

        3. REQUIRED SECTIONS: Identify EXACTLY what sections must be included in the proposal response. Include both main sections and any specified subsections. Use the exact section names from the RFP.

        4. TIMELINE: Extract all dates, deadlines, and milestones mentioned in the RFP.

        5. BUDGET CONSTRAINTS: Note any explicit budget limitations, pricing structures, or financial parameters mentioned.

        6. EVALUATION CRITERIA: Detail how the proposal will be scored or evaluated, including any weighted criteria.

        7. CLIENT PAIN POINTS: Identify specific problems or challenges the client is trying to solve, both explicit and implied.

        8. UNIQUE CONSIDERATIONS: Flag any special requirements, unusual constraints, or differentiating factors that stand out.

        Format your response as a structured analysis with clear headings for each category. Use bullet points for clarity. Extract specific, actionable information rather than general observations.

        RFP TEXT:
        {rfp_text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing RFP: {str(e)}")
            return f"Error analyzing RFP: {str(e)}"
    
    def extract_mandatory_criteria(self, rfp_analysis):
        """Extract mandatory criteria from RFP analysis"""
        try:
            requirements_start = rfp_analysis.find("KEY REQUIREMENTS") + len("KEY REQUIREMENTS")
            requirements_end = rfp_analysis.find("DELIVERABLES", requirements_start)
            requirements_text = rfp_analysis[requirements_start:requirements_end].strip()
            
            # Extract mandatory criteria (items that must be addressed)
            mandatory_criteria = []
            for line in requirements_text.split('\n'):
                if line.strip() and ("must" in line.lower() or "required" in line.lower()):
                    mandatory_criteria.append(line.strip())
            
            return mandatory_criteria
        except:
            return []
    
    def extract_weighted_criteria(self, rfp_analysis):
        """Extract weighted evaluation criteria from RFP analysis"""
        try:
            criteria_start = rfp_analysis.find("EVALUATION CRITERIA") + len("EVALUATION CRITERIA")
            criteria_end = rfp_analysis.find("CLIENT PAIN POINTS", criteria_start)
            criteria_text = rfp_analysis[criteria_start:criteria_end].strip()
            
            weighted_criteria = []
            for line in criteria_text.split('\n'):
                if line.strip():
                    # Assuming criteria are in the format "Criterion (Weight)"
                    match = re.match(r'^(.*?)(\s+\((\d+)%\))?', line.strip())
                    if match:
                        criterion = match.group(1).strip()
                        weight = int(match.group(3)) if match.group(3) else 100
                        weighted_criteria.append((criterion, weight))
            
            return weighted_criteria
        except:
            return []
    
    def extract_deadlines(self, rfp_analysis):
        """Extract deadlines from RFP analysis"""
        try:
            timeline_start = rfp_analysis.find("TIMELINE") + len("TIMELINE")
            timeline_end = rfp_analysis.find("\n\n", timeline_start)
            timeline_text = rfp_analysis[timeline_start:timeline_end].strip()
            
            # Extract dates and deadlines
            deadlines = []
            for line in timeline_text.split('\n'):
                if line.strip() and any(term in line.lower() for term in ["deadline", "date", "due"]):
                    deadlines.append(line.strip())
            
            return deadlines
        except:
            return []
    
    def extract_deliverables(self, rfp_analysis):
        """Extract deliverables from RFP analysis"""
        try:
            deliverables_start = rfp_analysis.find("DELIVERABLES") + len("DELIVERABLES")
            deliverables_end = rfp_analysis.find("\n\n", deliverables_start)
            deliverables_text = rfp_analysis[deliverables_start:deliverables_end].strip()
            
            # Extract concrete deliverables
            deliverables = []
            for line in deliverables_text.split('\n'):
                if line.strip():
                    deliverables.append(line.strip())
            
            return deliverables
        except:
            return []
    
    def assess_compliance(self, rfp_analysis, internal_capabilities):
        """Assess compliance with internal capabilities"""
        try:
            # Extract key requirements
            requirements_pattern = r"KEY REQUIREMENTS(.*?)DELIVERABLES"
            requirements_text = re.search(requirements_pattern, rfp_analysis, re.DOTALL)
            requirements_text = requirements_text.group(1).strip() if requirements_text else ""
            
            # Create prompt for compliance assessment
            prompt = f"""
            Assess compliance between RFP requirements and internal capabilities:

            RFP REQUIREMENTS:
            {requirements_text}

            INTERNAL CAPABILITIES:
            Technical: {', '.join(internal_capabilities.get('technical', []))}
            Functional: {', '.join(internal_capabilities.get('functional', []))}

            For each RFP requirement, determine if we can:
            - Fully comply
            - Partially comply
            - Cannot comply

            Flag any requirements where we cannot comply or have significant gaps. Provide specific explanations for each compliance status.

            Format your response as a structured markdown table with columns:
            | Requirement | Compliance Status | Explanation |
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error assessing compliance: {str(e)}")
            return "Error assessing compliance."
    
    def extract_required_sections(self, rfp_analysis):
        """Extract required sections from RFP analysis"""
        try:
            sections_start = rfp_analysis.find("REQUIRED SECTIONS") + len("REQUIRED SECTIONS")
            sections_end = rfp_analysis.find("\n\n", sections_start)
            sections_text = rfp_analysis[sections_start:sections_end].strip()
            sections = [s.strip() for s in sections_text.split("\n") if s.strip()]
            return sections
        except:
            return []
    
    def generate_section(self, section_name, rfp_analysis, rfp_section_content, client_background, differentiators, evaluation_criteria, relevant_kb_content, client_name):
        """Generate a proposal section with adjusted client specificity requirements"""
        requirements_pattern = r"KEY REQUIREMENTS(.*?)DELIVERABLES"
        section_specific_requirements = re.search(requirements_pattern, rfp_analysis, re.DOTALL)
        section_specific_requirements = section_specific_requirements.group(1).strip() if section_specific_requirements else ""
        
        # Filter and process knowledge base content
        kb_content_list = []
        for item in relevant_kb_content:
            score = item["score"]
            doc = item["document"]
            
            # Only include truly relevant content
            if score < 0.5:
                continue
            rel_indicator = "Very Relevant" if score > 0.8 else "Relevant"
            
            # Handle case studies differently for client specificity
            if section_name.lower() in ["case studies", "references", "past performance"]:
                content = doc["content"]
            else:
                # Remove specific client identifiers from reference materials
                content = doc["content"]
                if any(sensitive_term in content.lower() for sensitive_term in 
                    ["client", "customer", "company", "specific", "confidential", "proprietary"]):
                    content = "⚠️ CONTAINS SENSITIVE REFERENCES - DO NOT COPY DIRECTLY: " + content
            
            kb_content_list.append(f"--- {rel_indicator} REFERENCE CONCEPT ---\nTopic: {doc['section_name']}\n\n{content}\n")
        
        kb_content = "\n\n".join(kb_content_list[:3])  # Limit to top 3 most relevant references
        
        # Check if this is a pricing-related section
        is_pricing_section = any(term in section_name.lower() for term in ["commercial", "pricing", "cost", "financial", "budget", "price"])
        
        pricing_instruction = ""
        if not is_pricing_section:
            pricing_instruction = "DO NOT include any pricing information, costs, or financial details in this section. Pricing should only be discussed in the Commercial Proposal section."
        
        # Adjusted prompt with modified client specificity requirements
        prompt = f"""
        # STRATEGIC PROPOSAL SECTION GENERATION

        ## PRIMARY FOCUS
        Section to Create: "{section_name}"

        ## RFP ANALYSIS CONTEXT
        {rfp_analysis}

        ## CLIENT BACKGROUND
        {client_background}

        ## EVALUATION CRITERIA
        {evaluation_criteria}

        ## DIFFERENTIATORS
        {differentiators}

        ## REFERENCE MATERIAL
        Use the following reference material for inspiration:
        {kb_content}

        ## GENERATION INSTRUCTIONS
        1. Create content addressing the RFP requirements
        2. For most sections, incorporate client-specific details and examples
        3. For Case Studies/References sections, provide general examples that demonstrate capability
        4. Maintain professional tone throughout
        5. Avoid overly technical language unless specified in RFP
        6. Use markdown formatting for readability
        7. {pricing_instruction}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert proposal writer. Create content that balances client specificity with general applicability based on section type."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating section {section_name}: {str(e)}")
            return f"Error generating section {section_name}: {str(e)}"
        
    def validate_proposal_client_specificity(self, proposal_sections, client_name):
        """Validates that the proposal is sufficiently client-specific"""
        issues = []
        
        for section_name, content in proposal_sections.items():
            # Check for client name mentions
            client_name_count = content.lower().count(client_name.lower())
            content_length = len(content)
            
            # Expected client name mention frequency based on content length
            expected_mentions = max(3, content_length // 500)
            
            if client_name_count < expected_mentions:
                issues.append(f"Section '{section_name}' has insufficient client references ({client_name_count} found, {expected_mentions} expected)")
                
            # Check for generic language
            generic_phrases = [
                "our clients", "many organizations", "typical companies", 
                "best practices", "industry standards", "our approach",
                "our methodology", "our process", "our solution"
            ]
            
            for phrase in generic_phrases:
                if phrase in content.lower():
                    issues.append(f"Section '{section_name}' contains generic phrase: '{phrase}'")
                    
        return issues
    
    def refine_section(self, section_name, current_content, feedback, client_name):
        """Refine a section based on user feedback"""
        # Replace names with current client name
        current_content = current_content.replace("CLIENT_NAME", client_name)
        current_content = current_content.replace("COMPANY_NAME", "Your Company Name")
        
        prompt = f"""
        # SECTION REFINEMENT

        ## CURRENT SECTION CONTENT
        {current_content}

        ## USER FEEDBACK
        {feedback}

        ## REFINEMENT INSTRUCTIONS
        Revise the section to address the feedback provided. Maintain the professional tone and structure while incorporating the suggested improvements. If the feedback suggests adding specific information, ensure it's included in a relevant part of the section. If the feedback suggests restructuring, improve the organization while preserving all essential content.

        Provide the refined section content.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error refining section {section_name}: {str(e)}")
            return f"Error refining section {section_name}: {str(e)}"
    
    def generate_compliance_matrix(self, rfp_analysis):
        """Generate a compliance matrix using the new prompt"""
        key_requirements_pattern = r"KEY REQUIREMENTS(.*?)DELIVERABLES"
        key_requirements = re.search(key_requirements_pattern, rfp_analysis, re.DOTALL)
        key_requirements = key_requirements.group(1).strip() if key_requirements else ""
        
        prompt = f"""
        Create a comprehensive compliance matrix that maps RFP requirements to our proposal sections.

        Use the following RFP analysis to identify all requirements:
        {key_requirements}

        For each requirement:
        1. Quote the exact requirement language from the RFP
        2. Identify which proposal section(s) address this requirement
        3. Provide a brief (1-2 sentence) explanation of how our proposal addresses this requirement
        4. Indicate compliance status: "Fully Compliant", "Partially Compliant", or "Non-Compliant"

        Format the output as a structured markdown table with the following columns:
        | RFP Requirement | Reference | Addressing Section(s) | How Addressed | Compliance Status |

        Where "Reference" refers to the section/page number in the original RFP.

        Ensure every requirement from the KEY REQUIREMENTS section of the RFP analysis is included.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating compliance matrix: {str(e)}")
            return "Error generating compliance matrix."
    
    def perform_risk_assessment(self, rfp_analysis):
        """Generate a risk assessment using the new prompt"""
        prompt = f"""
        Create a comprehensive risk assessment for this proposal based on the following RFP analysis:
        {rfp_analysis}

        Identify specific risks in the following categories:

        1. TECHNICAL RISKS: Integration challenges, technology limitations, compatibility issues
        2. TIMELINE RISKS: Schedule constraints, dependencies, resource availability
        3. SCOPE RISKS: Unclear requirements, potential scope changes, feature creep
        4. CLIENT RELATIONSHIP RISKS: Communication challenges, alignment issues, expectation management
        5. DELIVERY RISKS: Quality assurance, testing limitations, deployment challenges
        6. EXTERNAL RISKS: Market conditions, regulatory issues, third-party dependencies

        For each identified risk, provide:
        1. A specific, concrete description of the risk
        2. Probability assessment (Low, Medium, High) with brief justification
        3. Impact assessment (Low, Medium, High) with brief justification
        4. Specific mitigation strategy including both preventive and contingency approaches
        5. Risk owner (which team or role should manage this risk)

        Format as a well-structured markdown table. Prioritize the top 2-3 risks in each category rather than creating an exhaustive list. Focus on risks specific to this client and project.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating risk assessment: {str(e)}")
            return "Error generating risk assessment."
    
    def research_client_background(self, client_name):
        """Research client background using the new prompt"""
        prompt = f"""
        Based on the client name '{client_name}', create a strategic client profile for proposal customization.

        The profile should include:

        1. ORGANIZATION OVERVIEW:
           - Industry position and primary business focus
           - Approximate size (employees, revenue if public)
           - Geographic presence and market focus
           - Key products or services

        2. STRATEGIC PRIORITIES:
           - Current business challenges or transformation initiatives
           - Recent technology investments or digital initiatives
           - Growth areas or new market entries
           - Corporate values or mission emphasis

        3. DECISION-MAKING CONTEXT:
           - Organizational structure relevant to this proposal
           - Likely stakeholders and their priorities
           - Previous vendor relationships or relevant partnerships
           - Procurement or decision-making approach if known

        4. TECHNOLOGY LANDSCAPE:
           - Current systems or platforms likely in use
           - Technology stack preferences if known
           - Prior implementation successes or challenges
           - Digital maturity assessment

        Focus on factual information that can be verified. Where specific details aren't available, provide industry-standard insights that would still be relevant. Include only information that would directly enhance proposal customization.

        Format the response with clear headings and concise bullet points for easy reference.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error researching client: {str(e)}")
            return "Client background information not available."
    
    def evaluate_proposal_alignment(self, evaluation_criteria, proposal_sections):
        """Evaluate proposal alignment using the new prompt"""
        prompt = f"""
        Evaluate how effectively our proposed sections align with the evaluation criteria identified in the RFP analysis.

        RFP EVALUATION CRITERIA:
        {evaluation_criteria}

        PROPOSED SECTIONS:
        {proposal_sections}

        For each evaluation criterion:

        1. Identify which specific proposal section(s) address this criterion
        2. Rate our coverage as:
           - STRONG: Comprehensively addresses all aspects of the criterion
           - ADEQUATE: Addresses core requirements but could be strengthened
           - NEEDS IMPROVEMENT: Insufficient coverage or missing key elements
           - ABSENT: Not addressed in current proposal structure

        3. Provide specific, actionable recommendations to strengthen our alignment, such as:
           - Content additions or emphasis changes
           - Supporting evidence or examples to include
           - Structural improvements or section reorganization
           - Cross-references between sections to reinforce key points

        Format as a structured markdown table with columns:
        | Evaluation Criterion | Addressing Section(s) | Coverage Rating | Recommendations |

        Conclude with 3-5 high-priority, specific actions that would most significantly improve our alignment with the evaluation criteria.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error evaluating proposal alignment: {str(e)}")
            return "Error evaluating proposal alignment with RFP criteria."
    
    def generate_executive_summary(self, client_background, rfp_analysis, differentiators, solution_overview, client_name):
        """Generate an executive summary using the specialized prompt"""
        prompt = f"""
        Create a compelling Executive Summary for this proposal based on the following inputs:

        CLIENT BACKGROUND:
        {client_background}

        RFP ANALYSIS:
        {rfp_analysis}

        KEY DIFFERENTIATORS:
        {differentiators}

        SOLUTION OVERVIEW:
        {solution_overview}

        Your Executive Summary should:

        1. Open with a concise statement acknowledging the client's specific needs and challenges
        2. Present our understanding of their primary objectives in pursuing this project
        3. Outline our proposed approach at a high level (without technical detail)
        4. Highlight 2-3 key differentiators that make our solution uniquely valuable
        5. Reference our relevant experience and qualifications specifically relevant to their needs
        6. Close with a compelling value proposition that addresses their business outcomes

        Keep the Executive Summary to approximately 500 words using clear, confident language that demonstrates both understanding and expertise. Avoid generic claims and focus on client-specific value. Use minimal formatting - short paragraphs with occasional bold text for emphasis.

        The Executive Summary should stand alone if separated from the full proposal while compelling the reader to continue.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating executive summary: {str(e)}")
            return f"Error generating executive summary: {str(e)}"
    
    def generate_full_proposal(self, rfp_text, client_name=None, company_info=None, template_sections=None):
        """Generate a full proposal with adjusted client specificity requirements"""
        print("Analyzing RFP...")
        rfp_analysis = self.analyze_rfp(rfp_text)
        
        # Extract required sections from analysis or use template sections
        if template_sections:
            required_sections = template_sections
        else:
            required_sections = self.extract_required_sections(rfp_analysis)
        
        # Research client background
        if client_name:
            client_background = self.research_client_background(client_name)
        else:
            client_background = "Client background not provided."
            
        # Identify differentiators
        if company_info:
            differentiators = company_info.get("differentiators", "Company differentiators not provided.")
        else:
            differentiators = "Company differentiators not provided."
            
        # Extract evaluation criteria from analysis
        criteria_pattern = r"EVALUATION CRITERIA(.*?)CLIENT PAIN POINTS"
        evaluation_criteria = re.search(criteria_pattern, rfp_analysis, re.DOTALL)
        evaluation_criteria = evaluation_criteria.group(1).strip() if evaluation_criteria else "Evaluation criteria not specified."
        
        # Generate all sections
        proposal_sections = {}
        for section_name in required_sections:
            print(f"Generating section: {section_name}")
            
            # Find relevant RFP section content
            rfp_sections = extract_sections_from_rfp(rfp_text)
            rfp_section_content = None
            for rfp_section in rfp_sections:
                if section_name.lower() in rfp_section.lower() or rfp_section.lower() in section_name.lower():
                    rfp_section_content = rfp_sections[rfp_section]
                    break
            
            # Expand query for better retrieval
            expanded_query = expand_query(section_name + " " + (rfp_section_content or ""))
            
            # Multi-hop search for better results
            relevant_kb_content = self.kb.multi_hop_search(expanded_query, k=3)
            
            # Generate section with adjusted client specificity
            proposal_sections[section_name] = self.generate_section(
                section_name,
                rfp_analysis,
                rfp_section_content,
                client_background,
                differentiators,
                evaluation_criteria,
                relevant_kb_content,
                client_name
            )
        
        # Add executive summary if needed
        if "Executive Summary" not in proposal_sections and client_name:
            print("Generating Executive Summary...")
            # Create a condensed view of key sections
            section_highlights = ""
            key_sections = ["Approach", "Methodology", "Solution", "Benefits", "Implementation"]
            for section in key_sections:
                matching_section = next((s for s in proposal_sections.keys() if section.lower() in s.lower()), None)
                if matching_section:
                    content_preview = proposal_sections[matching_section][:200] + "..."
                    section_highlights += f"## {matching_section} Preview\n{content_preview}\n\n"
            
            # Generate executive summary
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert proposal writer specializing in executive summaries."},
                        {"role": "user", "content": f"""
                        Create a concise executive summary for the proposal to {client_name}.
                        Highlight key differentiators and solution overview.
                        Keep it to approximately 500 words.
                        """}
                    ],
                    temperature=0.4
                )
                proposal_sections["Executive Summary"] = response.choices[0].message.content
            except Exception as e:
                print(f"Error generating Executive Summary: {str(e)}")
        
        return {
            "analysis": rfp_analysis,
            "sections": proposal_sections,
            "client_background": client_background,
            "differentiators": differentiators,
            "required_sections": required_sections
        }

    def perform_quality_assurance(self, proposal_sections, rfp_analysis):
        """Perform quality assurance checks on the proposal"""
        prompt = f"""
        # QUALITY ASSURANCE CHECK

        ## PROPOSAL SECTIONS
        {json.dumps(proposal_sections, indent=2)}

        ## RFP ANALYSIS
        {rfp_analysis}

        ## QUALITY ASSURANCE INSTRUCTIONS
        Perform comprehensive quality assurance checks on the proposal sections:

        1. LANGUAGE TONE:
           - Evaluate if the tone is professional and confident
           - Check for overly technical language that might confuse the client
           - Identify any overly casual or informal language

        2. GRAMMAR AND SPELLING:
           - Identify any grammatical errors
           - Check for spelling mistakes
           - Verify consistency in verb tenses and subject-verb agreement

        3. COMPLIANCE WITH RFP GUIDELINES:
           - Verify that all sections address the RFP requirements
           - Check if the proposal follows the structure requested in the RFP
           - Ensure all mandatory elements from the RFP are included

        4. CONTENT QUALITY:
           - Evaluate if claims are supported with evidence
           - Check for vague statements that should be more specific
           - Identify any sections that could benefit from additional details

        Provide specific feedback for each section, including:
        - Exact location of the issue
        - Description of the problem
        - Suggested improvement

        Format the output as a structured markdown report with the following sections:
        ## Overall Quality Score (1-10)
        ## Tone Assessment
        ## Grammar and Spelling Issues
        ## Compliance with RFP Guidelines
        ## Content Quality Feedback
        ## Actionable Improvements
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error performing quality assurance: {str(e)}")
            return "Error performing quality assurance."

    def generate_advanced_analysis(self, proposal_data, rfp_analysis, internal_capabilities, client_name):
        """Generate advanced analysis without executive summary"""
        analysis_results = {}
        
        # Generate Compliance Matrix
        compliance_matrix = self.generate_compliance_matrix(rfp_analysis)
        analysis_results["compliance_matrix"] = compliance_matrix
        
        # Generate Risk Assessment
        risk_assessment = self.perform_risk_assessment(rfp_analysis)
        analysis_results["risk_assessment"] = risk_assessment
        
        # Evaluate Proposal Alignment
        evaluation_criteria = re.search(
            r"EVALUATION CRITERIA(.*?)CLIENT PAIN POINTS",
            rfp_analysis,
            re.DOTALL
        ).group(1).strip() if rfp_analysis else "Evaluation criteria not specified."
        
        alignment_assessment = self.evaluate_proposal_alignment(
            evaluation_criteria,
            list(proposal_data["sections"].keys())
        )
        analysis_results["alignment_assessment"] = alignment_assessment
        
        # Compliance Assessment
        compliance_assessment = self.assess_compliance(rfp_analysis, internal_capabilities)
        analysis_results["compliance_assessment"] = compliance_assessment
        
        return analysis_results

    def analyze_vendor_proposal(self, vendor_proposal_text, rfp_analysis, client_name):
        """Analyze vendor proposal against RFP requirements with detailed factual comparison"""
        # Extract specific RFP requirements for comparison
        requirements_pattern = r"KEY REQUIREMENTS(.*?)DELIVERABLES"
        rfp_requirements = re.search(requirements_pattern, rfp_analysis, re.DOTALL)
        rfp_requirements = rfp_requirements.group(1).strip() if rfp_requirements else "No requirements found"
        
        # Extract evaluation criteria for scoring
        criteria_pattern = r"EVALUATION CRITERIA(.*?)CLIENT PAIN POINTS"
        evaluation_criteria = re.search(criteria_pattern, rfp_analysis, re.DOTALL)
        evaluation_criteria = evaluation_criteria.group(1).strip() if evaluation_criteria else "No criteria found"
        
        # Generate analysis prompt with detailed instructions
        analysis_prompt = f"""
        # DETAILED VENDOR PROPOSAL ANALYSIS

        ## ANALYSIS INSTRUCTIONS:
        1. Perform a comprehensive analysis of the vendor proposal against the provided RFP requirements
        2. Evaluate the proposal based on the defined scoring metrics and client-specific priorities
        3. Provide specific examples from both the RFP and proposal to support your analysis
        4. Format your response with clear headings for each analysis category

        ## SCORING METRICS (weights in parentheses):
        - Requirement Match (30%): How well the proposal addresses all RFP requirements
        - Compliance (25%): Adherence to RFP specifications and constraints
        - Quality (20%): Depth and relevance of proposed solutions
        - Alignment (15%): Match with client priorities and evaluation criteria
        - Risk (10%): Identified risks and mitigation strategies

        ## RFP REQUIREMENTS:
        {rfp_requirements}

        ## EVALUATION CRITERIA:
        {evaluation_criteria}

        ## VENDOR PROPOSAL:
        {vendor_proposal_text}

        ## ANALYSIS FORMAT:
        ### Overall Score (0-100)
        ### Requirement Matching:
        - Fully Addressed Requirements
        - Partially Addressed Requirements
        - Unaddressed Requirements
        ### Compliance Assessment:
        - Met Requirements
        - Partially Met Requirements
        - Unmet Requirements
        ### Quality Evaluation:
        - Strengths
        - Weaknesses
        ### Alignment with Client Priorities:
        - Well-Aligned Aspects
        - Misaligned Aspects
        ### Risk Assessment:
        - Identified Risks
        - Mitigation Strategies
        ### Sentiment Analysis:
        - Tone assessment (positive, neutral, negative)
        - Confidence level of the vendor
        ### Comparative Analysis:
        - How this proposal compares to typical industry standards
        - Competitive advantages/disadvantages

        Provide specific page/section references from the proposal for each assessment point.
        Calculate an overall score based on the weighted metrics.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.2  # Lower temperature for more factual analysis
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing vendor proposal: {str(e)}")
            return f"Error analyzing vendor proposal: {str(e)}"

    def identify_gaps_and_risks(self, vendor_proposal_text, rfp_requirements):
        """Use machine learning to identify gaps and risks in vendor responses"""
        try:
            # Vectorize text
            vectorizer = TfidfVectorizer()
            documents = [rfp_requirements, vendor_proposal_text]
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Calculate similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            
            # Identify gaps based on low similarity
            gaps = []
            if similarity < 0.7:  # Threshold for identifying gaps
                gaps = ["Low coverage of key requirements", "Mismatch in proposed solutions"]
            
            # Risk assessment (simplified example)
            risks = []
            if similarity < 0.5:
                risks = ["High risk of non-compliance", "Potential scope creep"]
            
            return gaps, risks
        except Exception as e:
            print(f"Error identifying gaps and risks: {str(e)}")
            return [], []

# Word export function
def export_to_word(proposal_data, company_name, client_name, output_path, company_logo_path=None):
    """Export the generated proposal to a professionally formatted Word document"""
    doc = Document()
    
    # Set document styles
    styles = doc.styles
    
    # Modify heading styles
    heading1 = styles['Heading 1']
    heading1.font.size = Pt(16)
    heading1.font.bold = True
    
    heading2 = styles['Heading 2']
    heading2.font.size = Pt(14)
    heading2.font.bold = True
    
    # Set document properties
    doc.core_properties.author = company_name
    doc.core_properties.title = f"Proposal for {client_name}"
    
    # Add title page
    if company_logo_path and os.path.exists(company_logo_path):
        doc.add_picture(company_logo_path, width=Inches(2.0))
        doc.add_paragraph()  # Add some space after logo
    
    title = doc.add_heading(f"Proposal for {client_name}", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add subtitle
    subtitle = doc.add_paragraph()
    subtitle_run = subtitle.add_run(f"Prepared by {company_name}")
    subtitle_run.font.size = Pt(14)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add date
    date_para = doc.add_paragraph()
    date_run = date_para.add_run(datetime.now().strftime("%B %d, %Y"))
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add a page break
    doc.add_page_break()
    
    # Add table of contents title
    doc.add_heading("Table of Contents", 1)
    
    # Generate table of contents
    toc = doc.add_paragraph()
    for idx, section_name in enumerate(proposal_data["sections"]):
        toc.add_run(f"{section_name}").bold = True
        toc.add_run(f"...{idx+3}\n")
    
    # Add page break after TOC
    doc.add_page_break()
    
    # Add each section with proper formatting
    for section_name, section_content in proposal_data["sections"].items():
        doc.add_heading(section_name, 1)
        
        lines = section_content.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('### '):
                doc.add_heading(line[4:].strip(), 3)
            elif line.startswith('## '):
                doc.add_heading(line[3:].strip(), 2)
            elif line.startswith('# '):
                doc.add_heading(line[2:].strip(), 1)
            elif line.startswith('- ') or line.startswith('* '):
                p = doc.add_paragraph(line[2:], style='List Bullet')
            elif re.match(r'^\d+\.\s', line):
                p = doc.add_paragraph(re.sub(r'^\d+\.\s', '', line), style='List Number')
            elif line.startswith('|') and i+1 < len(lines) and '|--' in lines[i+1]:
                table_rows = []
                table_rows.append(line)
                i += 1
                while i < len(lines) and lines[i].startswith('|'):
                    table_rows.append(lines[i])
                    i += 1
                if len(table_rows) > 2:
                    header = table_rows[0].split('|')[1:-1]
                    table = doc.add_table(rows=len(table_rows)-1, cols=len(header))
                    table.style = 'Table Grid'
                    for j, cell_text in enumerate(header):
                        table.cell(0, j).text = cell_text.strip()
                    for row_idx, row_text in enumerate(table_rows[2:]):
                        cells = row_text.split('|')[1:-1]
                        for j, cell_text in enumerate(cells):
                            if j < len(header):
                                table.cell(row_idx+1, j).text = cell_text.strip()
                i -= 1
            elif line:
                p = doc.add_paragraph(line)
            i += 1
        
        if section_name != list(proposal_data["sections"].keys())[-1]:
            doc.add_page_break()
    
    doc.save(output_path)
    
    return output_path

# Main Streamlit UI
def main():
    st.set_page_config(page_title="AI Proposal Generator", layout="wide", page_icon="📄")
    
    # Apply custom CSS
    st.markdown(""" 
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #003366;
    }
    .stButton button {
        background-color: #003366;
        color: white;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .section-card {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .sidebar-content {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    
    if 'knowledge_base' not in st.session_state:
        try:
            kb_dir = st.session_state.config["knowledge_base"]["directory"]
            embedding_model = st.session_state.config["knowledge_base"]["embedding_model"]
            device = "cpu"  # Use CPU to avoid device issues
            st.session_state.knowledge_base = ProposalKnowledgeBase(kb_dir, embedding_model, device=device)
        except Exception as e:
            st.error(f"Failed to initialize knowledge base: {str(e)}")
            st.session_state.knowledge_base = None
    
    if 'generator' not in st.session_state:
        openai_key = st.session_state.config["api_keys"]["openai_key"]
        if not openai_key:
            openai_key = os.environ.get("OPENAI_API_KEY", "")
        
        if openai_key:
            st.session_state.generator = EnhancedProposalGenerator(st.session_state.knowledge_base, openai_key)
        else:
            st.session_state.generator = None
    
    if 'rfp_text' not in st.session_state:
        st.session_state.rfp_text = ""
    
    if 'rfp_analysis' not in st.session_state:
        st.session_state.rfp_analysis = None
    
    if 'proposal_data' not in st.session_state:
        st.session_state.proposal_data = {
            "sections": {},
            "required_sections": [],
            "client_background": None,
            "differentiators": None
        }
    
    if 'client_background' not in st.session_state:
        st.session_state.client_background = None
    
    if 'differentiators' not in st.session_state:
        st.session_state.differentiators = None
    
    if 'advanced_analysis' not in st.session_state:
        st.session_state.advanced_analysis = {
            "compliance_matrix": None,
            "risk_assessment": None,
            "executive_summary": None,
            "alignment_assessment": None,
            "quality_assurance": None,
            "compliance_assessment": None
        }
    
    if 'template_created' not in st.session_state:
        st.session_state.template_created = False
    
    if 'template_sections' not in st.session_state:
        st.session_state.template_sections = []
    
    if 'rfp_response_analysis' not in st.session_state:
        st.session_state.rfp_response_analysis = None
    
    if 'vendor_analysis' not in st.session_state:
        st.session_state.vendor_analysis = None
    
    # Header
    st.title("🚀 AI-Powered Proposal Generator")
    
    # Main workflow tabs
    tabs = st.tabs(["📋 Upload RFP", "📝 Template Creation", "📊 Generate Proposal", "📤 Export", "🔍 Advanced Analysis", "🔍 Vendor Proposal Evaluation"])
    
    # Tab 1: Upload RFP
    with tabs[0]:
        st.header("Upload and Analyze RFP")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            uploaded_file = st.file_uploader("Upload RFP Document", type=["docx", "pdf", "txt", "md"])
            
            if uploaded_file is not None:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
                temp_file.write(uploaded_file.getvalue())
                temp_file.close()
                
                try:
                    rfp_text = process_rfp(temp_file.name)
                    st.session_state.rfp_text = rfp_text
                    st.success(f"Successfully processed {uploaded_file.name}")
                    
                    with st.expander("Preview RFP Content", expanded=False):
                        st.text_area("RFP Text", rfp_text, height=300)
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                finally:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("### 📝 Instructions")
            st.markdown("""
            1. Upload your RFP document (PDF, Word, or text)
            2. We'll extract and analyze the key requirements
            3. Click 'Analyze RFP' to get insights
            4. Proceed to the next tab to create your template
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.rfp_text:
                if st.button("Analyze RFP", type="primary"):
                    with st.spinner("Analyzing RFP..."):
                        if st.session_state.generator:
                            rfp_analysis = st.session_state.generator.analyze_rfp(st.session_state.rfp_text)
                            st.session_state.rfp_analysis = rfp_analysis
                            
                            # Extract required sections
                            required_sections = st.session_state.generator.extract_required_sections(rfp_analysis)
                            st.session_state.required_sections = required_sections
                            
                            # Extract mandatory criteria
                            mandatory_criteria = st.session_state.generator.extract_mandatory_criteria(rfp_analysis)
                            st.session_state.mandatory_criteria = mandatory_criteria
                            
                            # Extract deadlines
                            deadlines = st.session_state.generator.extract_deadlines(rfp_analysis)
                            st.session_state.deadlines = deadlines
                            
                            # Extract deliverables
                            deliverables = st.session_state.generator.extract_deliverables(rfp_analysis)
                            st.session_state.deliverables = deliverables
                            
                            # Assess compliance
                            internal_capabilities = st.session_state.config.get("internal_capabilities", {})
                            compliance_assessment = st.session_state.generator.assess_compliance(rfp_analysis, internal_capabilities)
                            st.session_state.compliance_assessment = compliance_assessment
                            
                            st.success("RFP Analysis Complete")
                            st.markdown("### Key Insights")
                            st.markdown("#### Mandatory Criteria")
                            if st.session_state.mandatory_criteria:
                                st.markdown("\n".join([f"- {item}" for item in st.session_state.mandatory_criteria]))
                            else:
                                st.markdown("No mandatory criteria found.")
                            
                            st.markdown("#### Deadlines")
                            if st.session_state.deadlines:
                                st.markdown("\n".join([f"- {item}" for item in st.session_state.deadlines]))
                            else:
                                st.markdown("No deadlines found.")
                            
                            st.markdown("#### Deliverables")
                            if st.session_state.deliverables:
                                st.markdown("\n".join([f"- {item}" for item in st.session_state.deliverables]))
                            else:
                                st.markdown("No deliverables found.")
                            
                            st.markdown("#### Compliance Assessment")
                            st.markdown(st.session_state.compliance_assessment)
                            
                            st.markdown("#### Full RFP Analysis")
                            st.write(rfp_analysis)
                        else:
                            st.error("OpenAI API key is not configured")
    
    # Tab 2: Template Creation
    with tabs[1]:
        st.header("Create Proposal Template")
        
        if not st.session_state.rfp_analysis:
            st.warning("Please upload and analyze an RFP first.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Template Sections")
                
                if st.session_state.required_sections:
                    st.markdown("#### Sections from RFP Analysis")
                    for section in st.session_state.required_sections:
                        if st.checkbox(section, key=f"rfp_section_{section}"):
                            if section not in st.session_state.template_sections:
                                st.session_state.template_sections.append(section)
                
                st.markdown("#### Add Custom Sections")
                new_section_name = st.text_input("New Section Name")
                
                if st.button("Add Section", type="primary"):
                    if new_section_name:
                        if new_section_name not in st.session_state.template_sections:
                            st.session_state.template_sections.append(new_section_name)
                            st.success(f"Section '{new_section_name}' added to template")
                        else:
                            st.warning("Section already exists in template")
                    else:
                        st.warning("Please provide a section name")
                
                if st.session_state.template_sections:
                    st.markdown("#### Current Template Sections")
                    for i, section in enumerate(st.session_state.template_sections):
                        col = st.columns(4)
                        with col[0]:
                            st.write(section)
                        with col[3]:
                            if st.button("Remove", key=f"remove_{i}") and i < len(st.session_state.template_sections):
                                st.session_state.template_sections.pop(i)
                                st.rerun()
                
                if st.button("Proceed to Generate Proposal", type="primary"):
                    st.session_state.template_created = True
                    st.rerun()
            
            with col2:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### 📝 Template Instructions")
                st.markdown("""
                1. Select sections from the RFP analysis
                2. Add custom sections as needed
                3. Review your template sections
                4. Click "Proceed" and move to the next tab to generate your proposal
                """)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Generate Proposal
    with tabs[2]:
        st.header("Generate Proposal")
        
        if not st.session_state.template_created:
            st.warning("Please create a template first.")
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Proposal Configuration")
                
                client_name = st.text_input("Client Name", "Client Organization")
                
                company_info = {
                    "name": st.session_state.config["company_info"]["name"],
                    "differentiators": st.text_area("Company Differentiators", 
                                                  "Enter key differentiators that make your company stand out")
                }
                
                if st.button("Generate Proposal", type="primary"):
                    with st.spinner("Generating proposal..."):
                        try:
                            if not st.session_state.generator:
                                raise Exception("OpenAI API key is not configured")
                            
                            # Generate proposal using template sections
                            proposal_data = st.session_state.generator.generate_full_proposal(
                                st.session_state.rfp_text, 
                                client_name,
                                company_info,
                                st.session_state.template_sections
                            )
                            st.session_state.proposal_data = proposal_data
                            st.success("Proposal generated successfully!")
                        except Exception as e:
                            st.error(f"Error generating proposal: {str(e)}")
        
            with col2:
                st.markdown("### Generation Controls")
                st.markdown("""
                The proposal generation process:
                1. Uses the RFP analysis to identify required sections
                2. Embeds the RFP analysis for similarity search
                3. Retrieves relevant content from the knowledge base
                4. Generates sections that strictly adhere to RFP requirements
                """)
            
            if st.session_state.proposal_data and st.session_state.proposal_data["sections"]:
                st.markdown("---")
                st.header("Proposal Preview")
                
                # Create a list for section selection
                section_names = list(st.session_state.proposal_data["sections"].keys())
                
                # Add tabs for each section
                section_tabs = st.tabs(section_names)
                
                for i, section_name in enumerate(section_names):
                    content = st.session_state.proposal_data["sections"][section_name]
                    with section_tabs[i]:
                        st.markdown(content)
                        
                        st.markdown("---")
                        col1, col2 = st.columns([3, 1])
                        
                        # Add feedback logic
                        with col1:
                            feedback = st.text_area(
                                "Provide feedback to improve this section:",
                                key=[f"feedback_{section_name}"]
                            )
                        
                        with col2:
                            if st.button("Update Section", key=f"update_{section_name}") and feedback:
                                try:
                                    with st.spinner(f"Updating '{section_name}' based on feedback..."):
                                        if not st.session_state.generator:
                                            raise Exception("OpenAI API key is not configured")
                                        
                                        # Generate refined content
                                        refined_content = st.session_state.generator.refine_section(
                                            section_name, 
                                            content, 
                                            feedback,
                                            client_name
                                        )
                                        
                                        # Update the proposal data
                                        st.session_state.proposal_data["sections"][section_name] = refined_content
                                        
                                        # Trigger UI refresh
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating section: {str(e)}")
                
            else:
                st.info("No sections available. Generate your proposal first.")
    
    # Tab 4: Export
    with tabs[3]:
        st.header("Export Your Proposal")
        
        if not st.session_state.proposal_data or not st.session_state.proposal_data["sections"]:
            st.warning("Please generate your proposal first.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Export Settings")
                
                company_name = st.text_input("Company Name", st.session_state.config["company_info"]["name"])
                client_name = st.text_input("Client Name", "Client")
                
                uploaded_logo = st.file_uploader("Upload Company Logo (optional)", type=["png", "jpg", "jpeg"])
                logo_path = None
                if uploaded_logo:
                    image = Image.open(uploaded_logo)
                    logo_path = "temp_logo.png"
                    image.save(logo_path)
                
                export_format = st.selectbox(
                    "Export Format",
                    ["Word Document (.docx)", "Markdown (.md)"]
                )
                
                if st.button("Export Proposal", type="primary"):
                    with st.spinner("Exporting proposal..."):
                        try:
                            if export_format == "Word Document (.docx)":
                                output_path = f"Proposal_for_{client_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.docx"
                                
                                final_path = export_to_word(
                                    st.session_state.proposal_data,
                                    company_name,
                                    client_name,
                                    output_path,
                                    logo_path
                                )
                                
                                with open(final_path, "rb") as file:
                                    st.download_button(
                                        label="Download Word Document",
                                        data=file,
                                        file_name=output_path,
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                    )
                            else:
                                md_content = f"# Proposal for {client_name}\n\n"
                                md_content += f"Prepared by {company_name}\n\n"
                                md_content += f"Date: {datetime.now().strftime('%B %d, %Y')}\n\n"
                                md_content += "## Table of Contents\n\n"
                                
                                for section_name in st.session_state.proposal_data["sections"]:
                                    md_content += f"- [{section_name}](#{section_name.lower().replace(' ', '-')})\n"
                                
                                md_content += "\n---\n\n"
                                
                                for section_name, content in st.session_state.proposal_data["sections"].items():
                                    md_content += f"# {section_name}\n\n"
                                    md_content += content + "\n\n---\n\n"
                                
                                st.download_button(
                                    label="Download Markdown File",
                                    data=md_content,
                                    file_name=f"Proposal_for_{client_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                                    mime="text/markdown"
                                )
                            
                            st.success("Proposal exported successfully!")
                            
                            if logo_path and os.path.exists(logo_path):
                                os.remove(logo_path)
                                
                        except Exception as e:
                            st.error(f"Error exporting proposal: {str(e)}")
            
            with col2:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("### 📤 Export Options")
                st.markdown("""
                - **Word Document (.docx)**: Professional document with formatting, headers, and your company logo
                - **Markdown (.md)**: Text-based format that preserves all content and can be easily edited
                """)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 5: Advanced Analysis
    with tabs[4]:
        st.header("Advanced Proposal Analysis")
        
        if not st.session_state.rfp_analysis:
            st.warning("Please upload and analyze an RFP first.")
        else:
            if not st.session_state.proposal_data or not st.session_state.proposal_data["sections"]:
                st.warning("Please generate your proposal first.")
            else:
                if st.button("Generate Advanced Analysis", type="primary"):
                    with st.spinner("Generating Advanced Analysis..."):
                        try:
                            if st.session_state.generator:
                                internal_capabilities = st.session_state.config.get("internal_capabilities", {})
                                advanced_analysis = st.session_state.generator.generate_advanced_analysis(
                                    st.session_state.proposal_data,
                                    st.session_state.rfp_analysis,
                                    internal_capabilities,
                                    client_name
                                )
                                st.session_state.advanced_analysis = advanced_analysis
                                st.success("Advanced Analysis Complete!")
                                # Trigger UI refresh
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error generating advanced analysis: {str(e)}")
                
                    # Tab 5: Advanced Analysis continued
                if st.session_state.advanced_analysis:
                    st.markdown("### Advanced Analysis Results")
                    
                    if st.session_state.advanced_analysis.get("compliance_matrix"):
                        st.markdown("#### Compliance Matrix")
                        st.markdown(st.session_state.advanced_analysis["compliance_matrix"])
                    
                    if st.session_state.advanced_analysis.get("risk_assessment"):
                        st.markdown("#### Risk Assessment")
                        st.markdown(st.session_state.advanced_analysis["risk_assessment"])
                
                    if st.session_state.advanced_analysis.get("executive_summary"):
                        st.markdown("#### Executive Summary")
                        st.markdown(st.session_state.advanced_analysis["executive_summary"])
                    
                    if st.session_state.advanced_analysis.get("alignment_assessment"):
                        st.markdown("#### Alignment Assessment")
                        st.markdown(st.session_state.advanced_analysis["alignment_assessment"])
                    
                    if st.session_state.advanced_analysis.get("quality_assurance"):
                        st.markdown("#### Quality Assurance Report")
                        st.markdown(st.session_state.advanced_analysis["quality_assurance"])
                    
                    if st.session_state.advanced_analysis.get("compliance_assessment"):
                        st.markdown("#### Compliance Assessment")
                        st.markdown(st.session_state.advanced_analysis["compliance_assessment"])
                
                # Perform quality assurance if button is pressed
                if st.button("Perform Quality Assurance", type="secondary"):
                    with st.spinner("Performing quality assurance..."):
                        if st.session_state.generator:
                            qa_report = st.session_state.generator.perform_quality_assurance(
                                st.session_state.proposal_data["sections"],
                                st.session_state.rfp_analysis
                            )
                            st.session_state.advanced_analysis["quality_assurance"] = qa_report
                            st.rerun()

    # Tab 6: Vendor Proposal Evaluation
    with tabs[5]:
        st.header("Vendor Proposal Evaluation")
        
        if not st.session_state.rfp_analysis:
            st.warning("Please upload and analyze an RFP first.")
        else:
            # File upload for vendor proposals
            uploaded_vendor_proposal = st.file_uploader("Upload Vendor Proposal", type=["docx", "pdf", "txt", "md"])
            
            if uploaded_vendor_proposal:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_vendor_proposal.name.split('.')[-1]}")
                temp_file.write(uploaded_vendor_proposal.getvalue())
                temp_file.close()
                
                try:
                    proposal_text = process_rfp(temp_file.name)
                    st.session_state.vendor_proposal_text = proposal_text
                    st.success(f"Successfully processed {uploaded_vendor_proposal.name}")
                    
                    with st.expander("Preview Proposal Content", expanded=False):
                        st.text_area("Proposal Text", proposal_text, height=300)
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                finally:
                    if os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)
            
            if hasattr(st.session_state, 'vendor_proposal_text') and st.session_state.vendor_proposal_text:
                # Use a unique key for the text_input
                client_name = st.text_input("Client Name", "Client Organization", key="client_name_input")
                
                if st.button("Analyze Proposal", type="primary"):
                    with st.spinner("Analyzing proposal..."):
                        try:
                            if not st.session_state.generator:
                                raise Exception("OpenAI API key is not configured")
                            
                            # Analyze vendor proposal using the RFP analysis
                            analysis = st.session_state.generator.analyze_vendor_proposal(
                                st.session_state.vendor_proposal_text,
                                st.session_state.rfp_analysis,
                                client_name
                            )
                            
                            # Identify gaps and risks using machine learning
                            weighted_criteria = st.session_state.generator.extract_weighted_criteria(st.session_state.rfp_analysis)
                            rfp_requirements = " ".join([c[0] for c in weighted_criteria])
                            
                            gaps, risks = st.session_state.generator.identify_gaps_and_risks(
                                st.session_state.vendor_proposal_text,
                                rfp_requirements
                            )
                            
                            # Combine analysis with gaps and risks
                            if gaps or risks:
                                analysis += "\n\n### Identified Gaps:\n" + "\n".join([f"- {gap}" for gap in gaps])
                                analysis += "\n\n### Identified Risks:\n" + "\n".join([f"- {risk}" for risk in risks])
                            
                            st.session_state.vendor_analysis = analysis
                            st.success("Analysis Complete!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error analyzing proposal: {str(e)}")
            
            if hasattr(st.session_state, 'vendor_analysis') and st.session_state.vendor_analysis:
                st.markdown("### Analysis Results")
                st.markdown(st.session_state.vendor_analysis)
                
                # Extract key metrics from analysis
                try:
                    match_score = re.search(r"match\s*score\s*:\s*(\d+)%", st.session_state.vendor_analysis, re.IGNORECASE)
                    compliance_score = re.search(r"compliance\s*score\s*:\s*(\d+)%", st.session_state.vendor_analysis, re.IGNORECASE)
                    quality_score = re.search(r"quality\s*score\s*:\s*(\d+)%", st.session_state.vendor_analysis, re.IGNORECASE)
                    
                    if match_score and compliance_score and quality_score:
                        scores = {
                            "Requirement Match": int(match_score.group(1)),
                            "Compliance": int(compliance_score.group(1)),
                            "Content Quality": int(quality_score.group(1))
                        }
                        
                        # Create a bar chart of the scores
                        fig, ax = plt.subplots()
                        ax.bar(scores.keys(), scores.values())
                        ax.set_ylabel('Score (%)')
                        ax.set_title('Proposal Analysis Scores')
                        st.pyplot(fig)
                except:
                    pass

if __name__ == "__main__":
    main()