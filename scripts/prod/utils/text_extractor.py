# General imports
import os
from typing import List, Dict, Any
import re

# Text extraction imports
import fitz  # PyMuPDF
import pdfplumber

# Configure logging
from logging_config import logger


class TextExtractor:
    def __init__(self, pdf_path:str, line_tol: float = 2.0, para_gap_factor:float = 1.2, min_words:int = 6):
        """ Initializes the TextExtractor with the path to the PDF file.
        Args:
            pdf_path (str): Path to the PDF file.
            line_tol (float): Tolerance for grouping words into lines.
            para_gap_factor (float): Factor to determine paragraph gaps based on line gaps.
            min_words (int): Minimum number of words required to keep a line.
        """
        logger.info(f"Initializing TextExtractor for {pdf_path}")
        self._pdf_path = pdf_path
        self._data = []
        self._doc_name = os.path.basename(pdf_path)
        self.line_tol = line_tol
        self.para_gap_factor = para_gap_factor
        self.min_words = min_words

    def _get_table_bboxes_pdfplumber(self, page_num:int):
        """ Extracts table bounding boxes from a specific page using pdfplumber.
        Returns a list of bounding boxes in the format [x0, y0, x1, y1]."""
        with pdfplumber.open(self._pdf_path) as pdf:
            page = pdf.pages[page_num]
            bboxes = [t.bbox for t in page.find_tables()]
        return bboxes

    def _group_words_to_paragraphs(self, words):
        """
        Groups words into visual lines, then merges lines into paragraphs using vertical gaps.
        Returns text with paragraphs separated by double newlines, not single newlines.
        """
        if not words:
            return ""
        words_sorted = sorted(words, key=lambda w: (w[1], w[0]))  # sort by y, then x
        # Step 1: Group words into lines
        lines = []
        current_line = []
        current_y = words_sorted[0][1]
        for w in words_sorted:
            if abs(w[1] - current_y) > self.line_tol:
                lines.append((current_y, current_line))
                current_line = [w]
                current_y = w[1]
            else:
                current_line.append(w)
        if current_line:
            lines.append((current_y, current_line))
        # Step 2: Compute vertical gaps between lines
        if len(lines) < 2:
            return " ".join(w[4] for w in words_sorted)
        gaps = [abs(lines[i+1][0] - lines[i][0]) for i in range(len(lines)-1)]
        # Use median gap as the typical line gap
        median_gap = sorted(gaps)[len(gaps) // 2] if len(gaps) % 2 == 1 else (sorted(gaps)[len(gaps) // 2 - 1] + sorted(gaps)[len(gaps) // 2]) / 2
        para_threshold = median_gap * self.para_gap_factor
        # Step 3: Merge lines into paragraphs
        paragraphs = []
        paragraph_lines = []
        for i, (y, line_words) in enumerate(lines):
            line_text = " ".join(w[4] for w in line_words)
            if i == 0:
                paragraph_lines.append(line_text)
                continue
            vertical_gap = abs(y - lines[i-1][0])
            if vertical_gap > para_threshold:
                paragraphs.append(" ".join(paragraph_lines))
                paragraph_lines = [line_text]
            else:
                paragraph_lines.append(line_text)
        if paragraph_lines:
            paragraphs.append(" ".join(paragraph_lines))
        return "\n\n".join(paragraphs)

    def extract_text_advanced(self):
        """
        Extracts text from the PDF using PyMuPDF and pdfplumber, filtering out table content.
        
        Returns: 
            a list of dictionaries with metadata for each page.
        """
        doc = fitz.open(self._pdf_path)
        doc_name = self._doc_name
        results = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            words = page.get_textpage().extractWORDS()  # returns list of [x0, y0, x1, y1, "word", block_no, line_no, word_no]
            words = page.get_text("words")  # returns [x0, y0, x1, y1, "word", block_no, line_no, word_no]
            # 2. Get table bboxes from pdfplumber
            table_bboxes = self._get_table_bboxes_pdfplumber(page_num)
            # 3. Filter words outside tables
            def word_in_table(word, bboxes):
                x0, y0, x1, y1 = word[:4]
                for bx0, by0, bx1, by1 in bboxes:
                    # overlap test
                    if not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1):
                        return True
                return False
            filtered_words = [w for w in words if not word_in_table(w, table_bboxes)]
            # 4. Reconstruct text (lines or paragraphs)
            filtered_words_sorted = sorted(filtered_words, key=lambda w: (w[1], w[0]))  # sort by y, then x

            page_text = self._group_words_to_paragraphs(filtered_words_sorted)

            results.append({
                "doc_name": doc_name,
                "page": page_num + 1,
                "text": page_text,
                "table_bboxes": table_bboxes,
                # Add more metadata useful for RAG pipeline
                "num_words": len(filtered_words_sorted),
                "page_width": page.rect.width,
                "page_height": page.rect.height,
                # "block_count": len(page.get_text("blocks")),
                # "line_count": len(text_lines),
                "page_number": page_num + 1,
                "extraction_method": "PyMuPDF+pdfplumber+group_words_to_paragraphs"
            })
        doc.close()
        self._data = results
        return results

    def _clean_text_bullet_points(self) -> List[Dict[str, Any]]:
        """
        Clean up text entries in a list of dictionaries by moving bullet points
        from the end of lines to the beginning.
        
        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing text data
            
        Returns:
            List[Dict[str, Any]]: Cleaned list with bullet points moved to line beginnings
        """
        cleaned_data = []
        
        for item in self._data:
            # Create a copy of the dictionary to avoid modifying the original
            cleaned_item = item.copy()
            
            if 'text' in cleaned_item and cleaned_item['text']:
                text = cleaned_item['text']
                lines = text.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    # Check if line ends with bullet point
                    if line.rstrip().endswith('•'):
                        # Remove bullet from end and add to beginning
                        line_without_bullet = line.rstrip()[:-1].rstrip()
                        if line_without_bullet:  # Only add bullet if there's remaining text
                            cleaned_line = f"• {line_without_bullet}"
                        else:
                            cleaned_line = "•"
                        cleaned_lines.append(cleaned_line)
                    else:
                        cleaned_lines.append(line)
                
                # Join lines back together
                cleaned_item['text'] = '\n'.join(cleaned_lines)
            
            cleaned_data.append(cleaned_item)
        
        logger.info(f"✅ Cleaned bullet points in {len(cleaned_data)} text entries")
        
        #self._data = cleaned_data
        return cleaned_data

    def _clean_text_merge_lowercase_lines(self) -> List[Dict[str, Any]]:
        """
        Clean up text entries in a list of dictionaries by merging lines that start 
        with lowercase letters to the previous line.
        
        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing text data
            
        Returns:
            List[Dict[str, Any]]: Cleaned list with lowercase-starting lines merged
        """
        
        cleaned_data = []
        
        for item in self._data:
            # Create a copy of the dictionary to avoid modifying the original
            cleaned_item = item.copy()
            
            if 'text' in cleaned_item and cleaned_item['text']:
                text = cleaned_item['text']
                lines = text.split('\n')
                merged_lines = []
                for line in lines:
                    stripped_line = line.strip()
                    
                    # Check if line starts with a lowercase letter (ignoring whitespace and common prefixes)
                    if (stripped_line and 
                        re.match(r'^[a-z]', stripped_line) and 
                        merged_lines):  # Only merge if there's a previous line
                        
                        # Merge with the previous line, adding a space if needed
                        previous_line = merged_lines[-1].rstrip()
                        if previous_line and not previous_line.endswith(' '):
                            merged_lines[-1] = previous_line + ' ' + stripped_line
                        else:
                            merged_lines[-1] = previous_line + stripped_line
                    else:
                        # Add line as-is
                        merged_lines.append(line)
                        merged_lines.append(line)
                
                # Join lines back together
                cleaned_item['text'] = '\n'.join(merged_lines)
            
            cleaned_data.append(cleaned_item)
        
        #self._data = cleaned_data

        logger.info(f"✅ Merged lowercase-starting lines in {len(cleaned_data)} text entries")

        return cleaned_data
    
    def _clean_text_remove_toc_and_headings(self) -> List[Dict[str, Any]]:
        """
        Clean up text entries by removing Table of Contents (TOC) and headings.
        
        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing text data
            
        Returns:
            List[Dict[str, Any]]: Cleaned list with TOC and headings removed
        """
        cleaned_data = []
        
        for item in self._data:
            # Create a copy of the dictionary to avoid modifying the original
            cleaned_item = item.copy()
            
            if 'text' in cleaned_item and cleaned_item['text']:
                text = cleaned_item['text']
                lines = text.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    stripped_line = line.strip()
                    
                    # Skip empty lines
                    if not stripped_line:
                        cleaned_lines.append(line)
                        continue
                    
                    # Enhanced TOC patterns
                    # Pattern 1: text followed by dots and page number
                    toc_pattern1 = r'^.+\.{3,}\d+\s*$'
                    
                    # Pattern 2: text followed by multiple dots/spaces and page number (like your example)
                    toc_pattern2 = r'^.+\s\.{6,}\s*\d+\s*$'
                    
                    # Pattern 3: text with dots/spaces mixture ending with page number
                    toc_pattern3 = r'^.+[\.\s]{6,}\d+\s*$'
                    
                    # Pattern 4: for uppercase headings followed by dots, page number, and optional trailing uppercase letters
                    # Pattern 4: matches uppercase headings, dots, page number, then optional trailing uppercase/section text
                    toc_pattern4 = r'^[A-ZÉÈÀÇÎÔÛÄËÏÖÜŒÆ’0-9\s]+\.{6,}\s*\d+\s+[A-ZÉÈÀÇÎÔÛÄËÏÖÜŒÆ’0-9\s\.]+$'

                    # Pattern for headings with numbering (e.g., "2. Terminologie / Définition / Abréviation")
                    heading_pattern = r'^\d+(\.\d+)*\.\s+.+$'
                    
                    # Pattern for section headings without trailing dots (e.g., "2.2. Abréviations")
                    section_pattern = r'^\d+(\.\d+)+\.\s+[A-Z].*$'
                    
                    # Check if line matches any of the patterns to remove
                    if (re.match(toc_pattern1, stripped_line) or 
                        re.match(toc_pattern2, stripped_line) or
                        re.match(toc_pattern3, stripped_line) or
                        re.match(toc_pattern4, stripped_line) or
                        re.match(heading_pattern, stripped_line) or
                        re.match(section_pattern, stripped_line)):
                        # Skip this line (don't add to cleaned_lines)
                        continue
                    
                    # Additional check for common TOC/heading indicators
                    # Lines that are mostly uppercase and short might be headings
                    if (len(stripped_line) < 100 and 
                        stripped_line.isupper() and 
                        not any(char.isdigit() for char in stripped_line[-10:]) and
                        '.' not in stripped_line[-10:]):
                        continue
                    
                    # Keep the line
                    cleaned_lines.append(line)
                
                # Join lines back together
                cleaned_item['text'] = '\n'.join(cleaned_lines)
            
            cleaned_data.append(cleaned_item)
        
        #self._data = cleaned_data

        logger.info(f"✅ Removed TOC and headings from {len(cleaned_data)} text entries")
        return cleaned_data
    
    def _detect_and_fix_missing_line_breaks(self) -> List[Dict[str, Any]]:
        """
        Detect and fix missing line breaks in text entries. A missing line break is detected
        when an uppercase letter followed by lowercase letters appears after a word that 
        doesn't end with punctuation (., !, ?, :, ;).
        
        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing text data
            
        Returns:
            List[Dict[str, Any]]: Cleaned list with missing line breaks added
        """
        cleaned_data = []
        
        for item in self._data:
            # Create a copy of the dictionary to avoid modifying the original
            cleaned_item = item.copy()
            
            if 'text' in cleaned_item and cleaned_item['text']:
                text = cleaned_item['text']
                
                # Pattern to detect missing line breaks:
                # - Word boundary
                # - Not preceded by punctuation (., !, ?, :, ;) or whitespace
                # - Followed by space(s)
                # - Then an uppercase letter followed by lowercase letter(s)
                pattern = r'([^\.\!\?\:\;\s])\s+([A-Z][a-z])'
                
                # Replace with the same content but add a line break
                # \1 captures the character before the space
                # \2 captures the uppercase+lowercase word
                fixed_text = re.sub(pattern, r'\1\n\2', text)
                
                # Additional pattern for cases where there might be multiple spaces
                # or where the sentence structure is more complex
                pattern2 = r'([a-z])\s+([A-Z][a-z]+(?:\s+[a-z]+)*)'
                
                # Apply the second pattern more carefully to avoid false positives
                # Only apply if the uppercase word is likely a sentence start
                def replacement_func(match):
                    before = match.group(1)
                    after = match.group(2)
                    
                    # Common words that often start sentences (in French)
                    sentence_starters = {
                        'Ce', 'Cette', 'Ces', 'Le', 'La', 'Les', 'Un', 'Une', 'Des',
                        'Il', 'Elle', 'Ils', 'Elles', 'On', 'Nous', 'Vous',
                        'Dans', 'Pour', 'Avec', 'Sans', 'Sous', 'Sur', 'Vers',
                        'Selon', 'Pendant', 'Après', 'Avant', 'Depuis',
                        'Ainsi', 'Alors', 'Aussi', 'Cependant', 'Donc', 'Enfin',
                        'Ensuite', 'Néanmoins', 'Puis', 'Toutefois',
                        'Lorsque', 'Quand', 'Si', 'Bien', 'Comme',
                        'Article', 'Chapitre', 'Section', 'Partie', 'Annexe', 'En'
                    }
                    
                    first_word = after.split()[0] if after.split() else after
                    
                    # If the word is likely a sentence starter, add line break
                    if first_word in sentence_starters or len(first_word) > 8:
                        return f'{before}\n{after}'
                    else:
                        return match.group(0)  # Return unchanged
                
                # Apply the more sophisticated pattern
                fixed_text = re.sub(pattern2, replacement_func, fixed_text)
                
                cleaned_item['text'] = fixed_text
            
            cleaned_data.append(cleaned_item)
        
        #self._data = cleaned_data


        logger.info(f"✅ Fixed missing line breaks in {len(cleaned_data)} text entries")
        return cleaned_data
    
    def _clean_text_remove_headers_footers(self) -> List[Dict[str, Any]]:
        """
        Clean up text entries by removing common headers and footers patterns.
        
        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing text data
            
        Returns:
            List[Dict[str, Any]]: Cleaned list with headers/footers removed
        """
        cleaned_data = []
        
        # Define patterns to remove
        patterns_to_remove = [
            r'INTERNE SNCF',
            r'Référentiel Ingénierie',
            r'Document propriété de la SNCF\. Tous droits réservés',
            r'INTERNE SNCF Document propriété de la SNCF\. Tous droits réservés',
            r'^\s*\d{3,5}\s*$',  # Any 3-5 digit number (e.g., page code)
            r'Version \d+ du \d{2}-\d{2}-\d{4}',       # "Version X du DD-MM-YYYY"
            r'Édition du \d{4}-\d{2}-\d{2} Version \d+ du \d{4}-\d{2}-\d{2}',
            r'Référentiel Direction de l\'Ingénierie Document d\'application',
            r'Etablissement des plans techniques - Directives et particularités propres aux postes à platines, postes à commande informatique et postes informatiques',
            r'Page \d+\s*$',
            r'^\s*Page \d+\s*$'
            r'Applicable dès réception',
            r'IG\d{5,}',  # General IG code pattern
            r'IN\d{5,}',  # General IN code pattern
            r'Référence-article\s*:\s*IG\d{5,}-\d{6,}-\d{2,}[A-Z]?',
            r'Référence-article\s*:\s*IN\d{5,}-\d{6,}-\d{2,}[A-Z]?',
            r'Émetteur\s*:\s*Département Signalisation.*DGII SF \d+',
        ]
        
        for item in self._data:
            # Create a copy of the dictionary to avoid modifying the original
            cleaned_item = item.copy()
            
            if 'text' in cleaned_item and cleaned_item['text']:
                text = cleaned_item['text']
                lines = text.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    stripped_line = line.strip()
                    
                    # Skip empty lines but preserve them for formatting
                    if not stripped_line:
                        cleaned_lines.append(line)
                        continue
                    
                    # Check if line matches any of the patterns to remove
                    should_remove = False
                    for pattern in patterns_to_remove:
                        if re.search(pattern, stripped_line, re.IGNORECASE):
                            should_remove = True
                            break
                    
                    # If line doesn't match any removal pattern, keep it
                    if not should_remove:
                        cleaned_lines.append(line)
                
                # Join lines back together
                cleaned_item['text'] = '\n'.join(cleaned_lines)
            
            cleaned_data.append(cleaned_item)

        #self._data = cleaned_data

        logger.info(f"✅ Removed headers/footers from {len(cleaned_data)} text entries")
        return cleaned_data
    
    def _clean_text_remove_short_lines_and_linebreaks(self) -> List[Dict[str, Any]]:
        """
        Removes lines with fewer than min_words and unnecessary line breaks from text entries.

        Returns:
            List[Dict[str, Any]]: Cleaned list with short lines and excessive line breaks removed
        """
        cleaned_data = []
        min_words = self.min_words

        for item in self._data:
            cleaned_item = item.copy()
            if 'text' in cleaned_item and cleaned_item['text']:
                lines = cleaned_item['text'].split('\n')
                # Remove lines with less than min_words (count only real words, not empty strings)
                filtered_lines = [line for line in lines if len([w for w in line.split() if w.strip()]) >= min_words]
                # Remove unnecessary consecutive line breaks
                text = "\n".join(filtered_lines)
                text = re.sub(r'\n{2,}', '\n', text)  # Replace multiple line breaks with a single one
                cleaned_item['text'] = text.strip()
            cleaned_data.append(cleaned_item)

        logger.info(f"✅ Removed short lines and unnecessary line breaks in {len(cleaned_data)} text entries")
        return cleaned_data
    
    def _remove_empty_text_entries(self) -> List[Dict[str, Any]]:
        """
        Remove dictionaries from the list where the 'text' entry is an empty string.
        
        Args:
            data (List[Dict[str, Any]]): List of dictionaries containing text data
            
        Returns:
            List[Dict[str, Any]]: Filtered list with non-empty 'text' entries only
        """
        filtered_data = [item for item in self._data if item.get('text', '').strip() != '']
        
        #self._data = cleaned_data

        logger.info(f"✅ Removed {len(self._data) - len(filtered_data)} empty text entries")
        return filtered_data
    
    def clean_text(self) -> List[Dict[str, Any]]:
        """
        Clean up text entries in the data by applying various cleaning methods.
        
        Returns:
            List[Dict[str, Any]]: Cleaned list of text entries
        """
        logger.info("Starting text cleaning process...")
        
        # Step 1: Remove headers and footers
        self._data = self._clean_text_remove_headers_footers()

        # Step 2: Remove TOC and headings
        self._data = self._clean_text_remove_toc_and_headings()
        
        # Step 3: Move bullet points to line beginnings
        self._data = self._clean_text_bullet_points()
        
        # Step 4: Detect and fix missing line breaks
        self._data = self._detect_and_fix_missing_line_breaks()
        
        # Step 5: Merge lowercase-starting lines
        self._data = self._clean_text_merge_lowercase_lines()
        
        # Step 6: Remove short lines and unnecessary line breaks
        self._data = self._clean_text_remove_short_lines_and_linebreaks()

        # Step 7: Remove empty text entries
        self._data = self._remove_empty_text_entries()
        
        logger.info("Text cleaning process completed.")

        return self._data
    
    def get_text(self) -> str:
        """
        Get the cleaned text from the data.
        
        Returns:
            str: Concatenated cleaned text from all entries
        """
        return "\n\n".join(item['text'] for item in self._data if 'text' in item and item['text'])
    
    def get_text_metadata(self) -> str:
        """
        Get the metadata for the cleaned text from the data.

        Returns:
            str: Concatenated metadata from all entries
        """
        return "\n\n".join(f"doc: {item['doc_name']} Page {item['page_num']}:\n{item['text']}\n\n"
                           for item in self._data if 'text' in item and item['text'])
    
    def get_data(self) -> List[Dict[str, Any]]:
        """
        Get the cleaned data with metadata.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries with cleaned text and metadata
        """
        return self._data
    
    def get_doc_name(self) -> str:
        """
        Get the document name.
        
        Returns:
            str: Name of the document
        """
        return self._doc_name