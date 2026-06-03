"""
Intelligent Internal Assessment Question Paper Generator
Using Generative AI, RAG, LLM, FAISS, and Gradio
"""

import os
import re
import tempfile
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Environment setup
from dotenv import load_dotenv
load_dotenv()

# Third-party imports
import gradio as gr
import pandas as pd
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from pypdf import PdfReader
from docx import Document
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

@dataclass
class AssessmentConfig:
    """Configuration for different assessments"""
    name: str
    units: List[str]
    part_a_count: int
    part_a_marks: int
    part_b_count: int
    part_b_marks: int
    part_c_count: int
    part_c_marks: int
    total_marks: int

class SyllabusProcessor:
    """Handles syllabus extraction from various file formats"""
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"TXT extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text from TXT: {str(e)}")
    
    @staticmethod
    def extract_from_csv(file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(file_path)
            # Convert entire CSV to text format
            text = df.to_string()
            return text
        except Exception as e:
            logger.error(f"CSV extraction failed: {str(e)}")
            raise Exception(f"Failed to extract text from CSV: {str(e)}")
    
    @staticmethod
    def extract_text(file_path: str, file_type: str) -> str:
        """Main extraction method based on file type"""
        extractors = {
            'pdf': SyllabusProcessor.extract_from_pdf,
            'docx': SyllabusProcessor.extract_from_docx,
            'txt': SyllabusProcessor.extract_from_txt,
            'csv': SyllabusProcessor.extract_from_csv
        }
        
        extractor = extractors.get(file_type.lower())
        if not extractor:
            raise Exception(f"Unsupported file format: {file_type}")
        
        return extractor(file_path)
    
    @staticmethod
    def identify_units(text: str) -> Dict[str, str]:
        """Identify and extract Unit I to Unit V content"""
        units = {}
        
        # Pattern for unit detection (flexible formatting)
        unit_patterns = [
            r'(?:UNIT|Unit|unit)\s*([IVXLCDM]+)',
            r'([IVXLCDM]+)\.\s*(?:UNIT|Unit|unit)',
            r'(?:MODULE|Module|module)\s*([IVXLCDM]+)'
        ]
        
        # Split text into sections
        lines = text.split('\n')
        current_unit = None
        unit_content = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check for unit headers
            found_unit = None
            for pattern in unit_patterns:
                match = re.search(pattern, line_stripped, re.IGNORECASE)
                if match:
                    roman_num = match.group(1)
                    if roman_num in ['I', 'II', 'III', 'IV', 'V']:
                        found_unit = roman_num
                        break
            
            if found_unit:
                # Save previous unit
                if current_unit and unit_content:
                    units[current_unit] = '\n'.join(unit_content)
                current_unit = found_unit
                unit_content = [line]
            elif current_unit:
                unit_content.append(line)
        
        # Save last unit
        if current_unit and unit_content:
            units[current_unit] = '\n'.join(unit_content)
        
        # Ensure all units are present
        expected_units = ['I', 'II', 'III', 'IV', 'V']
        for unit in expected_units:
            if unit not in units:
                units[unit] = ""
                logger.warning(f"Unit {unit} not found in syllabus")
        
        return units

class RAGPipeline:
    """Handles RAG implementation with FAISS"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.chunk_embeddings = []
    
    def chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        return np.array(embeddings).astype('float32')
    
    def build_index(self, syllabus_text: str):
        """Build FAISS index from syllabus text"""
        try:
            # Chunk the text
            self.chunks = self.chunk_text(syllabus_text)
            if not self.chunks:
                raise Exception("No content chunks generated from syllabus")
            
            # Create embeddings
            logger.info(f"Creating embeddings for {len(self.chunks)} chunks")
            embeddings = self.create_embeddings(self.chunks)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {str(e)}")
            raise Exception(f"RAG pipeline initialization failed: {str(e)}")
    
    def retrieve_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant context for a query"""
        if not self.index or not self.chunks:
            return ""
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        # Retrieve relevant chunks
        relevant_chunks = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
        
        return "\n\n".join(relevant_chunks)

class QuestionGenerator:
    """Handles LLM-based question generation"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.groq_client = None
        self.initialize_groq()
    
    def initialize_groq(self):
        """Initialize Groq client with API key"""
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise Exception("GROQ_API_KEY environment variable not found. Please set it before running.")
        
        self.groq_client = Groq(api_key=api_key)
        logger.info("Groq client initialized successfully")
    
    def generate_questions(self, config: AssessmentConfig, units_content: Dict[str, str], 
                          course_code: str = "", course_name: str = "") -> Dict[str, Any]:
        """Generate questions based on assessment configuration"""
        
        # Prepare context based on units
        context = self.prepare_context(config.units, units_content)
        
        if not context:
            raise Exception(f"No content available for units: {', '.join(config.units)}")
        
        # Generate Part A questions (2 marks each)
        part_a = self.generate_part_a_questions(config, context)
        
        # Generate Part B questions (12/13 marks each)
        part_b = self.generate_part_b_questions(config, context)
        
        # Generate Part C questions (16/15 marks each)
        part_c = self.generate_part_c_questions(config, context)
        
        return {
            'course_code': course_code,
            'course_name': course_name,
            'assessment_name': config.name,
            'duration': "2 Hours" if config.total_marks == 50 else "3 Hours",
            'total_marks': config.total_marks,
            'part_a': part_a,
            'part_b': part_b,
            'part_c': part_c,
            'part_a_count': config.part_a_count,
            'part_a_marks': config.part_a_marks,
            'part_b_count': config.part_b_count,
            'part_b_marks': config.part_b_marks,
            'part_c_count': config.part_c_count,
            'part_c_marks': config.part_c_marks
        }
    
    def prepare_context(self, units: List[str], units_content: Dict[str, str]) -> str:
        """Prepare context from specified units"""
        unit_texts = []
        for unit in units:
            if unit in units_content and units_content[unit]:
                unit_texts.append(f"Unit {unit}:\n{units_content[unit]}")
        
        return "\n\n".join(unit_texts)
    
    def generate_part_a_questions(self, config: AssessmentConfig, context: str) -> List[str]:
        """Generate Part A questions (short answer questions)"""
        prompt = f"""
You are an expert question paper setter for a university-level course.

Based on the following syllabus content, generate exactly {config.part_a_count} short answer questions worth {config.part_a_marks} marks each.

Syllabus Content:
{context[:8000]}

Requirements:
1. Questions should test basic understanding and recall (Bloom's Taxonomy levels 1-2)
2. Each question should be answerable in 2-3 lines
3. Questions must be technically correct and based ONLY on the provided syllabus
4. No duplicate questions
5. Format each question as a clear, standalone question

Generate exactly {config.part_a_count} questions. Number them 1 to {config.part_a_count}.
Return only the questions, one per line, without any additional text.
"""

        try:
            response = self.groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a university professor creating examination questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            questions_text = response.choices[0].message.content
            questions = [q.strip() for q in questions_text.split('\n') if q.strip() and not q.strip().startswith('Note')]
            
            # Ensure we have exactly the required number
            return questions[:config.part_a_count]
            
        except Exception as e:
            logger.error(f"Failed to generate Part A questions: {str(e)}")
            raise Exception(f"Question generation failed: {str(e)}")
    
    def generate_part_b_questions(self, config: AssessmentConfig, context: str) -> List[str]:
        """Generate Part B questions (long answer questions)"""
        prompt = f"""
You are an expert question paper setter for a university-level course.

Based on the following syllabus content, generate exactly {config.part_b_count} long answer questions worth {config.part_b_marks} marks each.

Syllabus Content:
{context[:8000]}

Requirements:
1. Questions should test understanding, application, and analysis (Bloom's Taxonomy levels 2-4)
2. Each question should require detailed answers of 1-2 paragraphs
3. Questions must be technically correct and based ONLY on the provided syllabus
4. No duplicate questions
5. Include scenarios, problems, or analytical questions where appropriate

Generate exactly {config.part_b_count} questions. Number them 1 to {config.part_b_count}.
Return only the questions, one per line, without any additional text.
"""

        try:
            response = self.groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a university professor creating examination questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=3000
            )
            
            questions_text = response.choices[0].message.content
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            return questions[:config.part_b_count]
            
        except Exception as e:
            logger.error(f"Failed to generate Part B questions: {str(e)}")
            raise Exception(f"Question generation failed: {str(e)}")
    
    def generate_part_c_questions(self, config: AssessmentConfig, context: str) -> List[str]:
        """Generate Part C questions (case study/application oriented)"""
        
        if config.name == "IA3":
            question_type = "case study/application-oriented question"
        else:
            question_type = "comprehensive/analytical question"
        
        prompt = f"""
You are an expert question paper setter for a university-level course.

Based on the following syllabus content, generate exactly {config.part_c_count} {question_type} worth {config.part_c_marks} marks.

Syllabus Content:
{context[:8000]}

Requirements:
1. For IA3: Create a case study that tests synthesis and evaluation (Bloom's Taxonomy levels 5-6)
2. For IA1/IA2: Create a comprehensive question that tests analysis and synthesis (Bloom's Taxonomy levels 4-5)
3. Questions must be application-oriented and based ONLY on the provided syllabus
4. Include real-world scenarios where appropriate
5. Questions must require critical thinking

Generate exactly {config.part_c_count} question(s). Number it/them appropriately.
Return only the question(s), without any additional text.
"""

        try:
            response = self.groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a university professor creating case study questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=2000
            )
            
            questions_text = response.choices[0].message.content
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            return questions[:config.part_c_count]
            
        except Exception as e:
            logger.error(f"Failed to generate Part C questions: {str(e)}")
            raise Exception(f"Question generation failed: {str(e)}")

class PDFGenerator:
    """Handles PDF generation for question papers"""
    
    @staticmethod
    def generate_question_paper(questions_data: Dict[str, Any], output_path: str):
        """Generate professional question paper PDF"""
        
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=72)
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=30,
            fontName='Helvetica-Bold'
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName='Helvetica-Bold'
        )
        
        part_style = ParagraphStyle(
            'PartStyle',
            parent=styles['Heading3'],
            fontSize=12,
            alignment=TA_LEFT,
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        question_style = ParagraphStyle(
            'QuestionStyle',
            parent=styles['Normal'],
            fontSize=10,
            alignment=TA_LEFT,
            spaceAfter=8,
            fontName='Helvetica'
        )
        
        story = []
        
        # Institute Header
        story.append(Paragraph("CHENNAI INSTITUTE OF TECHNOLOGY", title_style))
        story.append(Paragraph("(Autonomous)", heading_style))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("DEPARTMENT OF INFORMATION TECHNOLOGY", heading_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Assessment Details
        details = f"""
        <b>Course Code:</b> {questions_data.get('course_code', 'N/A')}<br/>
        <b>Course Name:</b> {questions_data.get('course_name', 'N/A')}<br/>
        <b>Assessment:</b> {questions_data['assessment_name']}<br/>
        <b>Duration:</b> {questions_data['duration']}<br/>
        <b>Maximum Marks:</b> {questions_data['total_marks']}
        """
        
        story.append(Paragraph(details, question_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Part A
        story.append(Paragraph("PART – A", part_style))
        story.append(Paragraph(f"({questions_data['part_a_count']} × {questions_data['part_a_marks']} = {questions_data['part_a_count'] * questions_data['part_a_marks']} Marks)", 
                              question_style))
        story.append(Spacer(1, 0.1*inch))
        
        for i, question in enumerate(questions_data['part_a'], 1):
            story.append(Paragraph(f"{i}. {question}", question_style))
            story.append(Spacer(1, 0.1*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Part B
        story.append(Paragraph("PART – B", part_style))
        story.append(Paragraph(f"({questions_data['part_b_count']} × {questions_data['part_b_marks']} = {questions_data['part_b_count'] * questions_data['part_b_marks']} Marks)", 
                              question_style))
        story.append(Spacer(1, 0.1*inch))
        
        for i, question in enumerate(questions_data['part_b'], 1):
            story.append(Paragraph(f"{i}. {question}", question_style))
            story.append(Spacer(1, 0.15*inch))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Part C
        story.append(Paragraph("PART – C", part_style))
        story.append(Paragraph(f"({questions_data['part_c_count']} × {questions_data['part_c_marks']} = {questions_data['part_c_count'] * questions_data['part_c_marks']} Marks)", 
                              question_style))
        story.append(Spacer(1, 0.1*inch))
        
        for i, question in enumerate(questions_data['part_c'], 1):
            story.append(Paragraph(f"{i}. {question}", question_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Instructions
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("<b>Instructions:</b>", part_style))
        instructions = """
        1. Answer all questions
        2. Write clearly and legibly
        3. Assume reasonable data if not provided
        4. Draw diagrams wherever necessary
        """
        story.append(Paragraph(instructions, question_style))
        
        # Build PDF
        doc.build(story)

class QuestionPaperApp:
    """Main Gradio application class"""
    
    def __init__(self):
        self.syllabus_processor = SyllabusProcessor()
        self.rag_pipeline = RAGPipeline()
        self.question_generator = None
        self.current_questions = None
        self.course_info = {"code": "", "name": ""}
        
        # Assessment configurations
        self.assessments = {
            "IA1": AssessmentConfig(
                name="Internal Assessment - I",
                units=["I"],
                part_a_count=5,
                part_a_marks=2,
                part_b_count=2,
                part_b_marks=12,
                part_c_count=1,
                part_c_marks=16,
                total_marks=50
            ),
            "IA2": AssessmentConfig(
                name="Internal Assessment - II",
                units=["II", "III"],
                part_a_count=5,
                part_a_marks=2,
                part_b_count=2,
                part_b_marks=12,
                part_c_count=1,
                part_c_marks=16,
                total_marks=50
            ),
            "IA3": AssessmentConfig(
                name="Internal Assessment - III",
                units=["I", "II", "III", "IV", "V"],
                part_a_count=10,
                part_a_marks=2,
                part_b_count=5,
                part_b_marks=13,
                part_c_count=1,
                part_c_marks=15,
                total_marks=100
            )
        }
    
    def extract_course_info(self, text: str) -> Tuple[str, str]:
        """Extract course code and name from syllabus text"""
        # Pattern for course code (e.g., CS1234, IT5678, etc.)
        code_pattern = r'([A-Z]{2,4}\s*\d{3,4})'
        code_match = re.search(code_pattern, text[:5000])  # Search in first 5000 chars
        
        # Pattern for course name (look for common patterns)
        name_patterns = [
            r'(?:Course|Subject)[\s:]+([A-Z][A-Z\s&]+?)(?:\n|$)',
            r'(?:Name|Title)[\s:]+([A-Z][A-Z\s&]+?)(?:\n|$)'
        ]
        
        course_code = code_match.group(1) if code_match else ""
        course_name = ""
        
        for pattern in name_patterns:
            name_match = re.search(pattern, text[:5000], re.IGNORECASE)
            if name_match:
                course_name = name_match.group(1).strip()
                break
        
        return course_code, course_name
    
    def process_syllabus(self, file) -> Tuple[str, str, Dict[str, str]]:
        """Process uploaded syllabus file"""
        if file is None:
            raise Exception("No file uploaded. Please upload a syllabus file.")
        
        # Get file extension
        file_name = file.name
        file_extension = file_name.split('.')[-1].lower()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        
        try:
            # Extract text
            syllabus_text = self.syllabus_processor.extract_text(tmp_path, file_extension)
            
            if not syllabus_text or len(syllabus_text.strip()) < 100:
                raise Exception("Extracted syllabus is empty or too short. Please check the file content.")
            
            # Identify units
            units = self.syllabus_processor.identify_units(syllabus_text)
            
            # Extract course info
            course_code, course_name = self.extract_course_info(syllabus_text)
            self.course_info = {"code": course_code, "name": course_name}
            
            # Build RAG pipeline
            self.rag_pipeline.build_index(syllabus_text)
            
            # Initialize question generator
            self.question_generator = QuestionGenerator(self.rag_pipeline)
            
            # Prepare preview info
            preview = f"✅ Syllabus processed successfully!\n\n"
            preview += f"📚 Total length: {len(syllabus_text)} characters\n"
            preview += f"📖 Units found: {[unit for unit in units if units[unit]]}\n"
            preview += f"🏷️ Course Code: {course_code or 'Not found'}\n"
            preview += f"📘 Course Name: {course_name or 'Not found'}\n"
            
            return preview, syllabus_text, units
            
        except Exception as e:
            logger.error(f"Syllabus processing failed: {str(e)}")
            raise
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
    
    def generate_question_paper(self, syllabus_file, assessment_type, syllabus_text, units_dict):
        """Generate question paper based on assessment type"""
        
        try:
            # Validate inputs
            if syllabus_file is None:
                return "❌ Please upload a syllabus file first.", None
            
            if not syllabus_text or syllabus_text.strip() == "":
                return "❌ No syllabus content found. Please upload a valid file.", None
            
            # Convert units string representation back to dict if needed
            if isinstance(units_dict, str):
                import ast
                units_dict = ast.literal_eval(units_dict)
            
            # Get assessment config
            config = self.assessments.get(assessment_type)
            if not config:
                return f"❌ Invalid assessment type: {assessment_type}", None
            
            # Check if required units are present
            missing_units = [unit for unit in config.units if unit not in units_dict or not units_dict[unit]]
            if missing_units:
                return f"❌ Missing content for units: {', '.join(missing_units)}. Please check syllabus.", None
            
            # Generate questions
            questions = self.question_generator.generate_questions(
                config, units_dict, 
                self.course_info['code'], 
                self.course_info['name']
            )
            
            self.current_questions = questions
            
            # Format preview
            preview = self.format_questions_preview(questions)
            
            return preview, questions
            
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            return f"❌ Error generating questions: {str(e)}", None
    
    def format_questions_preview(self, questions: Dict[str, Any]) -> str:
        """Format questions for preview display"""
        preview = f"""
# {questions['assessment_name']}

**Course Code:** {questions['course_code'] or 'N/A'}
**Course Name:** {questions['course_name'] or 'N/A'}
**Duration:** {questions['duration']}
**Maximum Marks:** {questions['total_marks']}

---

## PART – A
*({questions['part_a_count']} × {questions['part_a_marks']} = {questions['part_a_count'] * questions['part_a_marks']} Marks)*

"""
        for i, q in enumerate(questions['part_a'], 1):
            preview += f"{i}. {q}\n\n"
        
        preview += f"\n## PART – B\n*({questions['part_b_count']} × {questions['part_b_marks']} = {questions['part_b_count'] * questions['part_b_marks']} Marks)*\n\n"
        
        for i, q in enumerate(questions['part_b'], 1):
            preview += f"{i}. {q}\n\n"
        
        preview += f"\n## PART – C\n*({questions['part_c_count']} × {questions['part_c_marks']} = {questions['part_c_count'] * questions['part_c_marks']} Marks)*\n\n"
        
        for i, q in enumerate(questions['part_c'], 1):
            preview += f"{i}. {q}\n\n"
        
        return preview
    
    def download_pdf(self, questions_data):
        """Generate and provide PDF for download"""
        if not questions_data:
            return None
        
        try:
            # Create temporary PDF file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                pdf_path = tmp_file.name
            
            # Generate PDF
            PDFGenerator.generate_question_paper(questions_data, pdf_path)
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"PDF generation failed: {str(e)}")
            return None
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="AI Question Paper Generator", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 📝 AI Question Paper Generator using GenAI + RAG
            ### Intelligent Internal Assessment Question Paper Generation System
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input components
                    syllabus_file = gr.File(
                        label="📄 Upload Syllabus",
                        file_types=[".pdf", ".docx", ".txt", ".csv"],
                        type="binary"
                    )
                    
                    assessment_type = gr.Radio(
                        choices=["IA1", "IA2", "IA3"],
                        label="Select Assessment",
                        value="IA1",
                        info="IA1: Unit I | IA2: Units II & III | IA3: All Units"
                    )
                    
                    generate_btn = gr.Button("🎯 Generate Question Paper", variant="primary")
                    
                    download_btn = gr.Button("📥 Download PDF", variant="secondary")
                    
                    # Hidden components for data passing
                    syllabus_text_state = gr.State("")
                    units_state = gr.State({})
                    questions_state = gr.State(None)
                
                with gr.Column(scale=2):
                    # Output components
                    status_text = gr.Textbox(
                        label="Status",
                        lines=3,
                        interactive=False
                    )
                    
                    question_preview = gr.Markdown(
                        label="Question Paper Preview",
                        value="### Ready to generate question paper\n\nUpload syllabus and select assessment type to begin."
                    )
            
            # Event handlers
            syllabus_file.change(
                fn=self.process_syllabus,
                inputs=[syllabus_file],
                outputs=[status_text, syllabus_text_state, units_state]
            )
            
            generate_btn.click(
                fn=self.generate_question_paper,
                inputs=[syllabus_file, assessment_type, syllabus_text_state, units_state],
                outputs=[question_preview, questions_state]
            ).then(
                fn=lambda: "✅ Questions generated successfully! You can now download the PDF.",
                outputs=[status_text]
            )
            
            download_btn.click(
                fn=self.download_pdf,
                inputs=[questions_state],
                outputs=[gr.File(label="Download Question Paper")]
            )
            
            # Add footer
            gr.Markdown("""
            ---
            ### 📋 Instructions
            1. Upload syllabus in PDF, DOCX, TXT, or CSV format
            2. Select the assessment type (IA1/IA2/IA3)
            3. Click 'Generate Question Paper' to create questions
            4. Preview questions and download PDF
            5. Questions follow Bloom's Taxonomy and university standards
            """)
        
        return demo

def main():
    """Main entry point"""
    try:
        # Check for API key
        if not os.getenv('GROQ_API_KEY'):
            print("⚠️ Warning: GROQ_API_KEY environment variable not set.")
            print("Please set it using: export GROQ_API_KEY='your-api-key'")
            print("Or create a .env file with GROQ_API_KEY=your-api-key")
            
            # For development, you can set it here (not recommended for production)
            # os.environ['GROQ_API_KEY'] = 'your-api-key-here'
        
        # Create and launch app
        app = QuestionPaperApp()
        demo = app.create_interface()
        
        # Launch the app
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
        
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
