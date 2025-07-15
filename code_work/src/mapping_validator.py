"""
문항-이미지 매핑 검증 및 수동 보정 함수들
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pdf_image_extractor import extract_questions_and_images_from_pdf, save_mapping_to_json


def validate_mapping_results(mapping_result: Dict) -> pd.DataFrame:
    """
    매핑 결과를 검증하고 검증 리포트를 생성합니다.
    
    Args:
        mapping_result: extract_questions_and_images_from_pdf 결과
        
    Returns:
        pd.DataFrame: 검증 리포트
    """
    
    validation_data = []
    
    for page_info in mapping_result['page_info']:
        page_num = page_info['page_number']
        question_count = page_info['question_count']
        image_count = page_info['image_count']
        mapping_count = len(page_info['mapping'])
        
        # 검증 메트릭 계산
        mapping_rate = mapping_count / question_count if question_count > 0 else 0
        image_utilization = mapping_count / image_count if image_count > 0 else 0
        
        validation_data.append({
            'page_number': page_num,
            'question_count': question_count,
            'image_count': image_count,
            'mapped_count': mapping_count,
            'mapping_rate': mapping_rate,
            'image_utilization': image_utilization,
            'questions': ', '.join([str(q[0]) if isinstance(q, tuple) else str(q) for q in page_info['questions']]) if page_info['questions'] else '',
            'status': get_mapping_status(mapping_rate, image_utilization)
        })
    
    return pd.DataFrame(validation_data)


def get_mapping_status(mapping_rate: float, image_utilization: float) -> str:
    """
    매핑 상태를 평가합니다.
    
    Args:
        mapping_rate: 문항 매핑 비율
        image_utilization: 이미지 활용 비율
        
    Returns:
        str: 상태 ('Good', 'Warning', 'Error')
    """
    
    if mapping_rate >= 0.8 and image_utilization >= 0.5:
        return 'Good'
    elif mapping_rate >= 0.5:
        return 'Warning'
    else:
        return 'Error'


def extract_number_from_question_id(question_id: str) -> str:
    """
    question_id에서 숫자 부분을 추출합니다.
    예: 'question_15' -> '15'
    """
    import re
    match = re.search(r'question_(\d+)', question_id)
    return match.group(1) if match else ""


def create_question_id_from_number(number: str) -> str:
    """
    숫자로부터 question_id를 생성합니다.
    예: '15' -> 'question_15'
    """
    return f"question_{number}"


def create_manual_correction_interface(mapping_result: Dict, json_source_path: str = None) -> Dict:
    """
    매핑 결과를 수동으로 보정할 수 있는 GUI 인터페이스를 생성합니다.
    
    Args:
        mapping_result: 원본 매핑 결과
        json_source_path: JSON 파일의 원본 경로 (이미지 디렉토리 찾기용)
        
    Returns:
        Dict: 보정된 매핑 결과
    """
    
    class MappingCorrectionApp:
        def __init__(self, mapping_data, json_source_path=None):
            self.mapping_data = mapping_data
            self.json_source_path = json_source_path
            self.corrected_mapping = mapping_data['question_image_mapping'].copy()
            
            self.root = tk.Tk()
            self.root.title("문항-이미지 매핑 보정 도구")
            self.root.geometry("1200x800")
            
            self.setup_ui()
            
        def setup_ui(self):
            # 메인 프레임
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 좌측 패널: 매핑 리스트
            left_frame = ttk.LabelFrame(main_frame, text="문항-이미지 매핑")
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            
            # 정렬 컨트롤 섹션
            sort_frame = ttk.Frame(left_frame)
            sort_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(sort_frame, text="정렬:").pack(side=tk.LEFT)
            
            sort_asc_btn = ttk.Button(sort_frame, text="문항 ID ↑", width=12,
                                    command=lambda: self.sort_tree('asc'))
            sort_asc_btn.pack(side=tk.LEFT, padx=(5, 2))
            
            sort_desc_btn = ttk.Button(sort_frame, text="문항 ID ↓", width=12,
                                     command=lambda: self.sort_tree('desc'))
            sort_desc_btn.pack(side=tk.LEFT, padx=2)
            
            # 정렬 상태 표시
            self.sort_status_label = ttk.Label(sort_frame, text="", foreground="blue")
            self.sort_status_label.pack(side=tk.LEFT, padx=(10, 0))
            
            # 트리뷰로 매핑 표시
            self.tree = ttk.Treeview(left_frame, columns=('Question', 'Image'), show='tree headings')
            self.tree.heading('#0', text='ID (클릭하여 정렬)')
            self.tree.heading('Question', text='문항번호')
            self.tree.heading('Image', text='이미지 파일')
            
            # 헤더 클릭으로 정렬 기능
            self.tree.heading('#0', command=lambda: self.toggle_sort())
            
            # 정렬 상태 추적
            self.current_sort = 'asc'  # 'asc', 'desc', None
            
            # 매핑 데이터 로드
            for question_id, image_path in self.corrected_mapping.items():
                self.tree.insert('', 'end', text=question_id, 
                               values=(question_id, Path(image_path).name))
            
            self.tree.pack(fill=tk.BOTH, expand=True)
            
            # 우측 패널: 이미지 미리보기
            right_frame = ttk.LabelFrame(main_frame, text="이미지 미리보기")
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
            
            self.image_label = ttk.Label(right_frame, text="이미지를 선택하세요")
            self.image_label.pack(fill=tk.BOTH, expand=True)
            
            # 하단 버튼
            button_frame = ttk.Frame(self.root)
            button_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Button(button_frame, text="수정", command=self.edit_mapping).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="삭제", command=self.delete_mapping).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="추가", command=self.add_mapping).pack(side=tk.LEFT, padx=5)
            ttk.Button(button_frame, text="저장", command=self.save_changes).pack(side=tk.RIGHT, padx=5)
            
            # 이벤트 바인딩
            self.tree.bind('<<TreeviewSelect>>', self.on_select)
            
            # 초기 정렬 수행 (기본: 오름차순)
            self.sort_tree('asc')
            
        def extract_question_number(self, question_id):
            """question_id에서 숫자 부분을 추출하여 정렬용 키 반환"""
            import re
            match = re.search(r'question_(\d+)', question_id)
            if match:
                return int(match.group(1))
            return 0  # 숫자가 없으면 0으로 처리
        
        def sort_tree(self, order='asc'):
            """TreeView를 question_id로 정렬합니다."""
            # 현재 TreeView의 모든 항목 수집
            items = []
            for child in self.tree.get_children():
                item_data = self.tree.item(child)
                question_id = item_data['text']
                values = item_data['values']
                items.append((question_id, values))
            
            # 정렬 수행
            if order == 'asc':
                items.sort(key=lambda x: self.extract_question_number(x[0]))
                self.current_sort = 'asc'
                status_text = "오름차순 정렬됨 ↑"
            else:  # desc
                items.sort(key=lambda x: self.extract_question_number(x[0]), reverse=True)
                self.current_sort = 'desc'
                status_text = "내림차순 정렬됨 ↓"
            
            # TreeView 모든 항목 삭제 후 재추가
            for child in self.tree.get_children():
                self.tree.delete(child)
            
            for question_id, values in items:
                self.tree.insert('', 'end', text=question_id, values=values)
            
            # 정렬 상태 업데이트
            self.sort_status_label.config(text=status_text)
            
            print(f"📊 정렬 완료: {len(items)}개 항목 ({order})")
        
        def toggle_sort(self):
            """헤더 클릭시 정렬 순서를 토글합니다."""
            if self.current_sort == 'asc':
                self.sort_tree('desc')
            else:
                self.sort_tree('asc')

        def on_select(self, event):
            selection = self.tree.selection()
            if selection:
                item = self.tree.item(selection[0])
                question_id = item['text']
                if question_id in self.corrected_mapping:
                    image_path = self.corrected_mapping[question_id]
                    self.show_image(image_path)
        
        def show_image(self, image_path: str):
            try:
                image = Image.open(image_path)
                # 이미지 리사이즈
                image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo  # 참조 유지
            except Exception as e:
                self.image_label.configure(text=f"이미지 로드 실패: {str(e)}")
        
        def edit_mapping(self):
            """
            선택된 매핑을 수정합니다.
            🔧 기능: 잘못 연결된 문항-이미지 매핑을 올바르게 수정 (문항 번호 + 이미지)
            """
            selection = self.tree.selection()
            if not selection:
                messagebox.showwarning("선택 필요", "수정할 매핑을 선택해주세요.")
                return
            
            item = self.tree.item(selection[0])
            old_question_id = item['text']
            current_image = self.corrected_mapping.get(old_question_id, "")
            current_number = extract_number_from_question_id(old_question_id)
            
            # 수정 다이얼로그 창 생성
            edit_window = tk.Toplevel(self.root)
            edit_window.title(f"매핑 수정: {old_question_id}")
            edit_window.geometry("700x500")
            edit_window.transient(self.root)
            edit_window.grab_set()
            
            # 현재 매핑 정보 표시
            info_frame = ttk.Frame(edit_window)
            info_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(info_frame, text=f"기존 문항 ID: {old_question_id}", font=("Arial", 12, "bold")).pack(anchor=tk.W)
            ttk.Label(info_frame, text=f"현재 이미지: {Path(current_image).name if current_image else '없음'}").pack(anchor=tk.W)
            
            # 문항 번호 수정 섹션
            number_frame = ttk.LabelFrame(edit_window, text="문항 번호 수정")
            number_frame.pack(fill=tk.X, padx=10, pady=5)
            
            number_input_frame = ttk.Frame(number_frame)
            number_input_frame.pack(fill=tk.X, padx=5, pady=5)
            
            ttk.Label(number_input_frame, text="문항 번호:").pack(side=tk.LEFT)
            number_entry = ttk.Entry(number_input_frame, width=10)
            number_entry.insert(0, current_number)
            number_entry.pack(side=tk.LEFT, padx=(5, 10))
            
            ttk.Label(number_input_frame, text="(예: 1, 2, 3... 숫자만 입력)", 
                     foreground="gray").pack(side=tk.LEFT)
            
            # 이미지 선택 리스트
            list_frame = ttk.LabelFrame(edit_window, text="새 이미지 선택 (선택사항)")
            list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # 안내 메시지 추가
            info_label = ttk.Label(list_frame, 
                                 text="💡 이미지를 선택하지 않으면 기존 이미지가 유지됩니다", 
                                 foreground="blue")
            info_label.pack(anchor=tk.W, padx=5, pady=2)
            
            # 사용 가능한 이미지 목록 생성
            available_images = self.get_available_images()
            
            # 리스트박스와 스크롤바
            scroll_frame = ttk.Frame(list_frame)
            scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            scrollbar = ttk.Scrollbar(scroll_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            image_listbox = tk.Listbox(scroll_frame, yscrollcommand=scrollbar.set, height=10)
            image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=image_listbox.yview)
            
            # 이미지 목록 채우기
            for img_path in available_images:
                image_listbox.insert(tk.END, Path(img_path).name)
            
            # 미리보기 레이블
            preview_label = ttk.Label(scroll_frame, text="이미지를 선택하면 미리보기가 표시됩니다", 
                                    relief=tk.SUNKEN, width=30)
            preview_label.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
            
            def on_image_select(event):
                """이미지 선택시 미리보기 표시"""
                selection_idx = image_listbox.curselection()
                if selection_idx:
                    selected_image = available_images[selection_idx[0]]
                    try:
                        img = Image.open(selected_image)
                        img.thumbnail((200, 200), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        preview_label.configure(image=photo, text="")
                        preview_label.image = photo
                    except Exception as e:
                        preview_label.configure(text=f"미리보기 실패: {str(e)}")
            
            image_listbox.bind('<<ListboxSelect>>', on_image_select)
            
            # 버튼 프레임
            button_frame = ttk.Frame(edit_window)
            button_frame.pack(fill=tk.X, padx=10, pady=5)
            
            def apply_changes():
                """변경사항 적용 (문항 번호 + 이미지)"""
                # 새 문항 번호 검증
                new_number = number_entry.get().strip()
                if not new_number.isdigit():
                    messagebox.showerror("입력 오류", "문항 번호는 숫자만 입력해주세요.")
                    return
                
                # 새 question_id 생성
                new_question_id = create_question_id_from_number(new_number)
                
                # 중복 검사 (자기 자신 제외)
                if new_question_id != old_question_id and new_question_id in self.corrected_mapping:
                    messagebox.showerror("중복 오류", f"문항 번호 {new_number}가 이미 존재합니다.")
                    return
                
                # 이미지 선택 - 선택사항으로 변경
                selection_idx = image_listbox.curselection()
                if selection_idx:
                    new_image = available_images[selection_idx[0]]
                    image_changed = True
                else:
                    new_image = current_image  # 기존 이미지 유지
                    image_changed = False
                
                # 기존 매핑 삭제 (question_id가 변경된 경우)
                if new_question_id != old_question_id:
                    del self.corrected_mapping[old_question_id]
                
                # 새 매핑 추가
                self.corrected_mapping[new_question_id] = new_image
                
                # TreeView 업데이트 - 정렬 유지
                if new_question_id != old_question_id:
                    # 기존 항목 삭제 후 새 항목 추가하고 다시 정렬
                    self.tree.delete(selection[0])
                    self.tree.insert('', 'end', text=new_question_id, 
                                   values=(new_question_id, Path(new_image).name))
                    # 현재 정렬 순서 유지
                    self.sort_tree(self.current_sort)
                else:
                    # 같은 항목의 이미지만 업데이트
                    self.tree.item(selection[0], values=(new_question_id, Path(new_image).name))
                
                # 성공 메시지
                if new_question_id != old_question_id and image_changed:
                    messagebox.showinfo("수정 완료", 
                                      f"문항 번호: {old_question_id} → {new_question_id}\n"
                                      f"이미지: {Path(new_image).name}")
                elif new_question_id != old_question_id and not image_changed:
                    messagebox.showinfo("수정 완료", 
                                      f"문항 번호: {old_question_id} → {new_question_id}\n"
                                      f"이미지: 기존 유지 ({Path(new_image).name})")
                elif image_changed:
                    messagebox.showinfo("수정 완료", 
                                      f"{new_question_id}의 이미지가 {Path(new_image).name}으로 변경되었습니다.")
                else:
                    messagebox.showinfo("수정 완료", "변경사항이 적용되었습니다.")
                
                edit_window.destroy()
            
            def cancel_changes():
                """변경사항 취소"""
                edit_window.destroy()
            
            ttk.Button(button_frame, text="적용", command=apply_changes).pack(side=tk.RIGHT, padx=5)
            ttk.Button(button_frame, text="취소", command=cancel_changes).pack(side=tk.RIGHT, padx=5)
        
        def get_available_images(self):
            """사용 가능한 모든 이미지 파일 목록을 반환합니다."""
            available_images = []
            
            # JSON 파일 위치를 기반으로 이미지 디렉토리 계산
            if self.json_source_path:
                json_path = Path(self.json_source_path)
                images_dir = json_path.parent / "extracted_images"
                
                if images_dir.exists():
                    # 모든 PNG 이미지 파일 찾기
                    available_images = list(images_dir.glob("*.png"))
                    available_images = [str(img) for img in available_images]
                    print(f"🔍 이미지 디렉토리: {images_dir}")
                    print(f"📊 발견된 이미지: {len(available_images)}개")
                else:
                    print(f"⚠️ 이미지 디렉토리 없음: {images_dir}")
            
            # 대체 방법: 매핑 데이터에서 이미지 디렉토리 경로 가져오기
            if not available_images:
                images_dir = Path(self.mapping_data.get('extracted_images_dir', 'extracted_images'))
                
                if images_dir.exists():
                    available_images = list(images_dir.glob("*.png"))
                    available_images = [str(img) for img in available_images]
                else:
                    # 매핑 데이터에서 기존 이미지 경로들 수집
                    for question_id, image_path in self.mapping_data.get('question_image_mapping', {}).items():
                        if image_path and Path(image_path).exists():
                            parent_dir = Path(image_path).parent
                            if parent_dir.exists():
                                available_images.extend([str(img) for img in parent_dir.glob("*.png")])
                                break  # 한 번만 수집하면 됨
            
            return list(set(available_images))  # 중복 제거
            
        def add_mapping(self):
            """
            새로운 문항-이미지 매핑을 추가합니다.
            ➕ 기능: 자동 매핑에서 누락된 문항에 이미지를 수동으로 연결
            """
            # 추가 다이얼로그 창 생성
            add_window = tk.Toplevel(self.root)
            add_window.title("새 매핑 추가")
            add_window.geometry("900x600")  # 창 크기 더 증가 (700x550 → 900x600)
            add_window.minsize(800, 550)    # 최소 크기도 증가
            add_window.transient(self.root)
            add_window.grab_set()
            
            # 문항 ID 입력 섹션
            id_frame = ttk.LabelFrame(add_window, text="문항 정보")
            id_frame.pack(fill=tk.X, padx=10, pady=5)
            
            ttk.Label(id_frame, text="문항 번호:").pack(side=tk.LEFT, padx=5)
            question_entry = ttk.Entry(id_frame, width=10)
            question_entry.pack(side=tk.LEFT, padx=5)
            
            ttk.Label(id_frame, text="(예: 1, 2, 3...)").pack(side=tk.LEFT, padx=5)
            
            # 누락된 문항 표시
            missing_questions = self.find_missing_questions()
            if missing_questions:
                ttk.Label(id_frame, text=f"누락된 문항: {', '.join(missing_questions)}", 
                         foreground="red").pack(side=tk.RIGHT, padx=5)
            
            # 이미지 선택 섹션
            image_frame = ttk.LabelFrame(add_window, text="이미지 선택")
            image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # 이미지 목록과 미리보기를 나란히 배치 (비율 조정)
            content_frame = ttk.Frame(image_frame)
            content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 좌측: 이미지 목록 (고정 너비)
            list_frame = ttk.Frame(content_frame)
            list_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
            list_frame.configure(width=300)  # 고정 너비
            list_frame.pack_propagate(False)  # 크기 고정
            
            # 이미지 표시 옵션
            option_frame = ttk.Frame(list_frame)
            option_frame.pack(fill=tk.X, pady=(0, 5))
            
            show_all_var = tk.BooleanVar(value=False)
            show_all_check = ttk.Checkbutton(option_frame, 
                                           text="모든 이미지 표시 (사용된 이미지 포함)", 
                                           variable=show_all_var,
                                           command=lambda: self.update_image_list(image_listbox, available_images, show_all_var.get()))
            show_all_check.pack(anchor=tk.W)
            
            ttk.Label(list_frame, text="사용 가능한 이미지:").pack(anchor=tk.W)
            
            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            image_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=15)
            image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=image_listbox.yview)
            
            # 사용 가능한 이미지 로드
            available_images = self.get_available_images()
            unused_images = self.get_unused_images(available_images)
            
            # 초기에는 사용되지 않은 이미지만 표시
            current_images = unused_images
            for img_path in current_images:
                image_listbox.insert(tk.END, Path(img_path).name)
            
            # 우측: 미리보기 (확장 가능)
            preview_frame = ttk.LabelFrame(content_frame, text="🖼️ 미리보기")
            preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
            
            # 미리보기 레이블 크기 증가
            preview_label = tk.Label(preview_frame, text="이미지를 선택하세요", 
                                   relief=tk.SUNKEN, width=40, height=20,
                                   bg='white', fg='gray')
            preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            def on_image_select(event):
                """이미지 선택시 미리보기 표시"""
                selection_idx = image_listbox.curselection()
                if selection_idx:
                    # 현재 표시된 이미지 목록에서 선택
                    if show_all_var.get():
                        selected_image = available_images[selection_idx[0]]
                    else:
                        current_unused = self.get_unused_images(available_images)
                        selected_image = current_unused[selection_idx[0]]
                    
                    try:
                        img = Image.open(selected_image)
                        # 미리보기 이미지 크기 증가 (250 → 400)
                        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        preview_label.configure(image=photo, text="")
                        preview_label.image = photo
                    except Exception as e:
                        preview_label.configure(text=f"미리보기 실패:\n{str(e)}")
            
            image_listbox.bind('<<ListboxSelect>>', on_image_select)
            
            # 버튼 섹션 - 창 하단에 고정
            # 구분선 추가
            separator = ttk.Separator(add_window, orient='horizontal')
            separator.pack(side=tk.BOTTOM, fill=tk.X, padx=10)
            
            button_frame = ttk.Frame(add_window)
            button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
            
            print("🔧 [DEBUG] 버튼 프레임 생성됨 (하단 고정, 구분선 포함)")  # 디버그 메시지
            
            def apply_addition():
                """새 매핑 추가 적용"""
                print("🔧 [DEBUG] 추가 버튼 클릭됨")  # 디버그 메시지
                question_num = question_entry.get().strip()
                selection_idx = image_listbox.curselection()
                
                if not question_num:
                    messagebox.showerror("입력 오류", "문항 번호를 입력해주세요.")
                    return
                
                if not selection_idx:
                    messagebox.showerror("선택 오류", "이미지를 선택해주세요.")
                    return
                
                question_id = f"question_{question_num}"
                
                # 중복 확인
                if question_id in self.corrected_mapping:
                    if not messagebox.askyesno("중복 확인", 
                                             f"{question_id}가 이미 존재합니다. 덮어쓰시겠습니까?"):
                        return
                
                # 매핑 추가
                # 현재 표시된 이미지 목록에서 선택된 이미지 가져오기
                if show_all_var.get():
                    selected_image = available_images[selection_idx[0]]
                else:
                    current_unused = self.get_unused_images(available_images)
                    selected_image = current_unused[selection_idx[0]]
                
                self.corrected_mapping[question_id] = selected_image
                
                # 트리뷰에 추가 (중복이면 업데이트) - 정렬 유지
                existing_item = None
                for item in self.tree.get_children():
                    if self.tree.item(item)['text'] == question_id:
                        existing_item = item
                        break
                
                if existing_item:
                    self.tree.item(existing_item, values=(question_id, Path(selected_image).name))
                else:
                    self.tree.insert('', 'end', text=question_id, 
                                   values=(question_id, Path(selected_image).name))
                    # 새 항목 추가 후 정렬 유지
                    self.sort_tree(self.current_sort)
                
                messagebox.showinfo("추가 완료", 
                                  f"{question_id}와 {Path(selected_image).name}이 매핑되었습니다.")
                add_window.destroy()
            
            def cancel_addition():
                """추가 취소"""
                add_window.destroy()
            
            # 버튼들을 더 크고 명확하게 만들기
            add_btn = ttk.Button(button_frame, text="✅ 추가", command=apply_addition)
            add_btn.pack(side=tk.RIGHT, padx=5, pady=5, ipadx=10)
            
            cancel_btn = ttk.Button(button_frame, text="❌ 취소", command=cancel_addition)
            cancel_btn.pack(side=tk.RIGHT, padx=5, pady=5, ipadx=10)
            
            print("🔧 [DEBUG] 추가/취소 버튼 생성 완료 (아이콘 포함)")  # 디버그 메시지
        
        def find_missing_questions(self):
            """현재 매핑에서 누락된 문항 번호들을 찾습니다."""
            existing_numbers = []
            for question_id in self.corrected_mapping.keys():
                number = extract_number_from_question_id(question_id)
                if number.isdigit():
                    existing_numbers.append(int(number))
            
            if not existing_numbers:
                return []
            
            # 1부터 최대값까지 중에서 누락된 번호 찾기
            max_num = max(existing_numbers)
            all_numbers = set(range(1, max_num + 1))
            existing_set = set(existing_numbers)
            missing = sorted(all_numbers - existing_set)
            
            return [str(num) for num in missing]
        
        def get_unused_images(self, available_images):
            """사용되지 않은 이미지들을 반환합니다."""
            used_images = set(self.corrected_mapping.values())
            unused = [img for img in available_images if img not in used_images]
            return unused

        def update_image_list(self, listbox, available_images, show_all=False):
            """이미지 목록을 업데이트합니다."""
            listbox.delete(0, tk.END)
            
            if show_all:
                current_images = available_images
                print(f"📊 모든 이미지 표시: {len(current_images)}개")
            else:
                current_images = self.get_unused_images(available_images)
                print(f"📊 사용되지 않은 이미지: {len(current_images)}개")
            
            for img_path in current_images:
                listbox.insert(tk.END, Path(img_path).name)
            
            return current_images

        def delete_mapping(self):
            selection = self.tree.selection()
            if selection:
                item = self.tree.item(selection[0])
                question_id = item['text']
                
                if messagebox.askyesno("삭제 확인", f"{question_id} 매핑을 삭제하시겠습니까?"):
                    del self.corrected_mapping[question_id]
                    self.tree.delete(selection[0])
        
        def save_changes(self):
            self.root.quit()
        
        def run(self):
            self.root.mainloop()
            return self.corrected_mapping
    
    # GUI 실행
    app = MappingCorrectionApp(mapping_result, json_source_path)
    corrected_mapping = app.run()
    
    # 결과 업데이트
    mapping_result['question_image_mapping'] = corrected_mapping
    return mapping_result


def generate_mapping_report(mapping_result: Dict, output_path: str = "mapping_report.html") -> None:
    """
    매핑 결과 리포트를 HTML로 생성합니다.
    
    Args:
        mapping_result: 매핑 결과
        output_path: 리포트 저장 경로
    """
    
    validation_df = validate_mapping_results(mapping_result)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>문항-이미지 매핑 리포트</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .good {{ background-color: #d4edda; }}
            .warning {{ background-color: #fff3cd; }}
            .error {{ background-color: #f8d7da; }}
            .summary {{ background-color: #e9ecef; padding: 15px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>문항-이미지 매핑 결과 리포트</h1>
        
        <div class="summary">
            <h3>요약</h3>
            <ul>
                <li>총 페이지 수: {len(mapping_result['page_info'])}</li>
                <li>총 매핑 수: {len(mapping_result['question_image_mapping'])}</li>
                <li>평균 매핑률: {validation_df['mapping_rate'].mean():.2%}</li>
                <li>평균 이미지 활용률: {validation_df['image_utilization'].mean():.2%}</li>
            </ul>
        </div>
        
        <h3>페이지별 상세 정보</h3>
        <table>
            <tr>
                <th>페이지</th>
                <th>문항 수</th>
                <th>이미지 수</th>
                <th>매핑 수</th>
                <th>매핑률</th>
                <th>이미지 활용률</th>
                <th>상태</th>
                <th>문항 목록</th>
            </tr>
    """
    
    for _, row in validation_df.iterrows():
        status_class = row['status'].lower()
        html_content += f"""
            <tr class="{status_class}">
                <td>{row['page_number']}</td>
                <td>{row['question_count']}</td>
                <td>{row['image_count']}</td>
                <td>{row['mapped_count']}</td>
                <td>{row['mapping_rate']:.2%}</td>
                <td>{row['image_utilization']:.2%}</td>
                <td>{row['status']}</td>
                <td>{row['questions']}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>매핑 세부 사항</h3>
        <table>
            <tr>
                <th>문항 ID</th>
                <th>이미지 파일</th>
            </tr>
    """
    
    for question_id, image_path in mapping_result['question_image_mapping'].items():
        html_content += f"""
            <tr>
                <td>{question_id}</td>
                <td>{Path(image_path).name}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>추출 로그</h3>
        <ul>
    """
    
    for log in mapping_result['extraction_log']:
        html_content += f"<li>{log}</li>"
    
    html_content += """
        </ul>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"리포트가 생성되었습니다: {output_path}")


def export_mapping_to_dataframe(mapping_result: Dict) -> pd.DataFrame:
    """
    매핑 결과를 pandas DataFrame으로 변환합니다.
    
    Args:
        mapping_result: 매핑 결과
        
    Returns:
        pd.DataFrame: 매핑 데이터프레임
    """
    
    mapping_data = []
    
    for question_id, image_path in mapping_result['question_image_mapping'].items():
        # 문항번호 추출
        question_num = question_id.replace('question_', '')
        
        # 페이지 정보 찾기
        page_info = None
        for page in mapping_result['page_info']:
            if question_id in page.get('mapping', {}):
                page_info = page
                break
        
        mapping_data.append({
            'question_id': question_id,
            'question_number': question_num,
            'image_path': image_path,
            'image_filename': Path(image_path).name,
            'page_number': page_info['page_number'] if page_info else None,
            'has_image': True
        })
    
    return pd.DataFrame(mapping_data)


# 메인 실행 함수
def main_mapping_workflow(pdf_path: str) -> pd.DataFrame:
    """
    전체 매핑 워크플로우를 실행합니다.
    
    Args:
        pdf_path: 처리할 PDF 파일 경로
        
    Returns:
        pd.DataFrame: 최종 매핑 결과
    """
    
    print(f"🔄 PDF 처리 시작: {pdf_path}")
    
    # PDF 파일명에서 타입 추출 (예: "3_raw_data_fromkoroad_사진형.pdf" -> "사진형")
    pdf_stem = Path(pdf_path).stem
    if "_" in pdf_stem:
        file_type = pdf_stem.split('_')[-1]
    else:
        file_type = pdf_stem
    
    # 출력 디렉토리 설정: G:\내 드라이브\4_paper\72_2_license_llm_how\data\results\{파일타입}\
    base_data_dir = Path(pdf_path).parent.parent / "data"
    output_dir = base_data_dir / "results" / file_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 출력 디렉토리: {output_dir}")
    
    # 1. 매핑 추출 (이미지도 해당 폴더에 저장)
    images_dir = output_dir / "extracted_images"
    mapping_result = extract_questions_and_images_from_pdf(pdf_path, str(images_dir))
    
    # 2. 검증 리포트 생성
    validation_df = validate_mapping_results(mapping_result)
    print("\n📊 검증 결과:")
    print(validation_df[['page_number', 'question_count', 'mapped_count', 'mapping_rate', 'status']])
    
    # 3. 리포트 생성 (해당 폴더에 저장)
    report_path = output_dir / f"mapping_report_{pdf_stem}.html"
    generate_mapping_report(mapping_result, str(report_path))
    
    # 4. DataFrame 변환
    final_df = export_mapping_to_dataframe(mapping_result)
    
    # 5. 결과 저장 (해당 폴더에 저장)
    csv_path = output_dir / f"question_image_mapping_{pdf_stem}.csv"
    final_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 6. JSON 결과도 저장
    json_path = output_dir / f"mapping_results_{file_type}.json"
    save_mapping_to_json(mapping_result, str(json_path))
    
    print(f"\n✅ 처리 완료:")
    print(f"   - 매핑 수: {len(final_df)}")
    print(f"   - CSV 저장: {csv_path}")
    print(f"   - 리포트: {report_path}")
    print(f"   - JSON 저장: {json_path}")
    print(f"   - 이미지 저장: {images_dir}")
    
    return final_df
