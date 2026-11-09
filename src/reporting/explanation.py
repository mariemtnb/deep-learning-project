"""
Write explanation-simple.docx: a plain-language guide to the interface and report.

Run:  python -m src.cli explanation
"""
from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
# The pipeline CLI sets DL_OUTPUTS_DIR from the config (paths.outputs).
OUT = Path(os.environ.get("DL_OUTPUTS_DIR", ROOT / "outputs"))
ACCENT = RGBColor(0x1F, 0x6F, 0x8B)

S = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
E = json.loads((OUT / "examples.json").read_text(encoding="utf-8"))

CONDITIONS = [
    ("Infiltration", "Something that should not be there (fluid, cells) has spread into the lung."),
    ("Effusion", "Fluid has collected around the lung."),
    ("Atelectasis", "Part of the lung has collapsed or is not filling with air."),
    ("Nodule", "A small round spot in the lung."),
    ("Mass", "A larger spot or lump, over about 3 cm."),
    ("Pneumothorax", "Air has leaked outside the lung, so the lung is pressed inward."),
    ("Consolidation", "Part of the lung is filled with fluid or other material instead of air."),
    ("Pleural thickening", "The thin lining around the lung has become thicker, often scarred."),
    ("Cardiomegaly", "The heart looks bigger than normal."),
    ("Emphysema", "The tiny air sacs are damaged, often from smoking, making breathing harder."),
    ("Edema", "Too much fluid inside the lungs, often linked to heart problems."),
    ("Fibrosis", "Scar tissue in the lung, which makes it stiff."),
    ("Pneumonia", "A lung infection."),
    ("Hernia", "An organ pushes through the wall that should hold it, here seen on the chest image."),
]

WORDS = [
    ("Dataset", "A collection of data, like one big folder of files or one big spreadsheet."),
    ("Row and column", "In a table, one row is one item (one X-ray) and one column is one fact about it (age)."),
    ("Label", "The correct answer attached to an example, which a computer learns from."),
    ("Model", "A program that learns patterns from examples instead of following written rules."),
    ("Training", "The stage where a model studies examples and adjusts itself."),
    ("Data cleaning", "Finding and fixing mistakes in the data."),
    ("Feature", "One useful piece of information given to a model."),
    ("Feature engineering", "Creating new useful information from what you already have."),
    ("Pipeline", "A fixed series of steps the data passes through, like a factory line."),
    ("Data leakage", "When a model accidentally sees test answers while learning, so its score looks better than it really is."),
    ("Median", "The middle value when numbers are sorted. Very high or low values barely affect it."),
    ("Outlier", "A value very far from all the others, such as an age of 148."),
    ("Imbalance", "When some categories have far more examples than others."),
    ("Bounding box", "A rectangle drawn on an image to show where a problem is."),
]


def shade(cell, color: str) -> None:
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(fill)


def picture(data_uri: str) -> io.BytesIO:
    return io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))


class Guide:
    def __init__(self) -> None:
        self.doc = Document()
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.left_margin = sec.right_margin = Cm(2.3)
        sec.top_margin = sec.bottom_margin = Cm(2.2)
        st = self.doc.styles
        st["Normal"].font.name = "Calibri"
        st["Normal"].font.size = Pt(11.5)
        st["Normal"].paragraph_format.space_after = Pt(7)
        st["Normal"].paragraph_format.line_spacing = 1.2
        for name, size in (("Title", 28), ("Heading 1", 18), ("Heading 2", 13)):
            st[name].font.name = "Calibri"
            st[name].font.size = Pt(size)
            st[name].font.color.rgb = ACCENT
            st[name].element.rPr.rFonts.set(qn("w:asciiTheme"), "")
            st[name].element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
            st[name].element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")

    def h1(self, text): self.doc.add_heading(text, 1)
    def h2(self, text): self.doc.add_heading(text, 2)
    def p(self, text): self.doc.add_paragraph(text)

    def bullets(self, items, style="List Bullet"):
        for item in items:
            self.doc.add_paragraph(item, style=style)

    def table(self, head, rows, widths):
        t = self.doc.add_table(rows=1, cols=len(head))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, text in enumerate(head):
            cell = t.rows[0].cells[i]
            cell.text = ""
            cell.paragraphs[0].add_run(text).bold = True
            shade(cell, "E4F0F4")
        trp = t.rows[0]._tr.get_or_add_trPr()
        trp.append(OxmlElement("w:tblHeader"))
        for row in rows:
            cells = t.add_row().cells
            for i, text in enumerate(row):
                cells[i].text = str(text)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
                for para in row.cells[i].paragraphs:
                    para.paragraph_format.space_after = Pt(2)
        self.doc.add_paragraph()

    def images(self, items, width=7.2):
        t = self.doc.add_table(rows=2, cols=len(items))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, (uri, caption) in enumerate(items):
            t.rows[0].cells[i].paragraphs[0].add_run().add_picture(picture(uri), width=Cm(width))
            t.rows[1].cells[i].text = caption
        self.doc.add_paragraph()


def n(value) -> str:
    return f"{value:,}"


def build() -> None:
    g = Guide()
    g.doc.add_paragraph("The chest X-ray project, explained simply", style="Title")
    g.p("A guide to the dashboard and the data tools. You do not need any medical "
        "or data science background to read it.")

    g.h1("1. The short version")
    g.p("Doctors read chest X-rays to find lung and heart problems. There are many images "
        "and it takes experts a lot of time. This project prepares the data needed to teach "
        "a computer to help them by flagging what may be wrong in an X-ray, using the image "
        "together with basic patient details such as age and sex.")
    g.p("So far, the data has been prepared. No AI model has been trained yet, which means "
        "nothing in the dashboard is a diagnosis or a prediction. Every number describes the "
        "data itself. A tool like this would support a doctor and would not replace one.")

    g.h1("2. Starting the interface")
    g.p("Open a terminal in the project folder and run this command:")
    g.doc.add_paragraph("python app/run.py", style="Intense Quote")
    g.p("Then open http://localhost:8000 in your browser. The interface runs only on your "
        "computer. The left menu has two parts: Data tools and Report.")

    g.h1("3. Data tools: add, treat, download")
    g.h2("Add a dataset")
    g.bullets([
        "From a link: paste a direct link to a file and press Download. The interface shows "
        "the download progress. A ZIP file is unpacked for you.",
        "From your computer: press Upload a file.",
        "Accepted tables are CSV, TSV, JSON and Parquet. Other files, such as images, are "
        "stored and can be downloaded, but only tables can be treated.",
    ])
    g.p("The link must lead straight to the file. If it opens a web page, or a site such as "
        "Kaggle asks you to sign in, you will see a message. In that case, download the file "
        "in your browser and use Upload instead. Each link is limited to 5 GB, and links to "
        "private or local network addresses are refused for safety.")
    g.h2("Look at a dataset")
    g.p("Pick a dataset from the list and click a table file. You will see:")
    g.table(["What you see", "What it means"], [
        ("Rows", "How many items the table holds."),
        ("Columns", "How many facts are recorded about each item."),
        ("Duplicate rows", "Rows that are exact copies of another row."),
        ("Missing cells", "Blank places where a value should be."),
        ("Type", "The kind of value: whole number, decimal, or text."),
        ("Unique", "How many different values a column holds."),
        ("Min, mean, max", "The smallest, average and largest number in a column."),
    ], [5, 11.2])
    g.h2("Treat the data")
    g.p("Tick the treatments you want, then press Apply. Your original file is never changed. "
        "The result is saved as a new file ending in __treated_1.csv, and the interface "
        "tells you exactly what changed.")
    g.table(["Treatment", "What it does", "When to use it"], [
        ("Remove duplicate rows", "Deletes exact copies.", "Almost always, unless repeats are real."),
        ("Fill missing values", "Numbers get the median. Text gets the most common value.", "When you would rather keep rows that have a few blanks."),
        ("Remove rows with missing values", "Deletes any row with a blank.", "When few rows are affected. It can delete a lot."),
        ("Trim text", "Removes stray spaces, so \"Male \" matches \"Male\".", "Almost always safe."),
        ("Cap extreme values", "Pulls very unusual numbers back to a limit.", "Carefully. A rare value can be real and important."),
        ("Rescale numbers to 0..1", "Puts every number on the same scale.", "Just before training a model."),
    ], [4.6, 6.3, 5.3])
    g.p("Cap extreme values and Rescale act on every numeric column, including ID numbers and "
        "yes/no columns stored as 0 and 1. Check the result before using it.")
    g.h2("Download")
    g.p("Each file has a Download link. Download all gives every file of a dataset in one ZIP. "
        "Delete removes an added dataset for good. The project dataset cannot be deleted.")

    g.h1("4. What the data is")
    g.h2("Dataset 1: NIH chest X-rays")
    g.p(f"The main dataset comes from the US National Institutes of Health. It holds "
        f"{n(S['n_images'])} X-ray images from {n(S['n_patients'])} patients. Each image comes "
        f"with a table row giving age, sex, how the picture was taken, and the findings. "
        f"One image can show several problems at once.")
    g.p("The findings were not written image by image by doctors for this dataset. A computer "
        "extracted them from the doctors' written reports, and about 1 in 10 is estimated to be "
        "wrong. Any model trained on them will inherit some of those mistakes.")
    g.h2("Dataset 2: Pneumonia X-rays")
    pn = S["pneumonia_dataset"]
    g.p(f"The second dataset has {n(sum(pn.values()))} X-rays of children, each marked either "
        f"normal or pneumonia. It is simpler, and it is separate: its patients are not the "
        f"same people as in the NIH dataset.")
    g.h2("Example images")
    nih = E["nih"]
    g.images([(nih[0]["img"], "No finding: nothing was reported on this X-ray."),
              (nih[1]["img"], "Cardiomegaly: the report says the heart is enlarged.")])
    if E.get("bbox"):
        g.images([(E["bbox"]["img"], "A bounding box: the orange rectangle marks where the "
                                     f"problem is. Only {n(S['has_bbox'])} images have one.")])
    g.p("How to look at an X-ray: dense parts such as bone appear white, and air appears dark. "
        "Healthy lungs are mostly dark. White patches inside the lungs can signal fluid or "
        "other problems. The image is shown as if you are facing the patient.")
    g.h2("The 14 conditions, in everyday words")
    counts = {k.replace("_", " ").lower(): v for k, v in S["pathology_counts"].items()}
    g.table(["Condition", "In simple words", "Images"],
            [(a, b, n(counts[a.lower()])) for a, b in CONDITIONS], [3.8, 10.2, 2.2])
    g.p("These are simplified descriptions to help you read the project. They are not medical advice.")

    g.h1("5. Reading the charts")
    nv = S["normal_vs_abnormal"]
    total = nv["normal"] + nv["abnormal"]
    single = S["multilabel_counts"]["1"]
    g.table(["Chart", "What it shows", "What to take away"], [
        ("Pathology prevalence", "How many images have each condition.",
         f"Very uneven: {n(S['pathology_counts']['Infiltration'])} for Infiltration against "
         f"{n(S['pathology_counts']['Hernia'])} for Hernia. A model will struggle with rare conditions."),
        ("Normal vs abnormal", "Images with no finding against those with at least one.",
         f"{nv['normal'] / total:.0%} are normal, so a model could score well by often saying "
         "normal. Accuracy alone can mislead."),
        ("Findings per image", "How many conditions appear together.",
         f"{n(single)} images have one finding, and many others have several, so the model "
         "must be able to give several answers."),
        ("Patient split", "How the images are divided for learning and testing.",
         f"Train {n(S['split_counts']['train'])}, validation {n(S['split_counts']['val'])}, "
         f"test {n(S['split_counts']['test'])}."),
        ("Age", "Images per age group.", f"The median age is {S['age']['median_clean']}. Children and over 75s are less represented."),
        ("Gender", "Images per sex.", f"{n(S['gender_counts']['M'])} male and {n(S['gender_counts']['F'])} female images, so men are more represented."),
        ("View position", "How the X-ray was taken.",
         f"PA ({n(S['view_counts']['PA'])}): standing, beam through the back. AP "
         f"({n(S['view_counts']['AP'])}): beam through the front, often for very ill patients "
         "in bed. The view itself can hint at how sick a patient is."),
        ("Pneumonia dataset", "Normal and pneumonia images per group.",
         f"Training has about three times more pneumonia than normal images, and validation "
         f"has only {pn['val/NORMAL'] + pn['val/PNEUMONIA']} images, which is very small."),
    ], [3.5, 5.2, 7.5])

    g.h1("6. What the pipeline did")
    g.p("Think of a kitchen. Raw data arrives like groceries, and the pipeline washes, sorts "
        "and prepares it before cooking, which here means training a model.")
    g.table(["Step", "What happened", "Status"], [
        ("Collect", "Read the NIH table and located every image. All "
                    f"{n(S['n_images_on_disk'])} images were found.", "Done"),
        ("Clean", f"Fixed damaged column names and repaired {S['age']['n_invalid_repaired']} "
                  "impossible ages, such as 148, using the median.", "Done"),
        ("Create features", "Turned the findings text into 14 yes/no columns, and added age "
                            "group, number of findings and follow-up visit.", "Done"),
        ("Split by patient", "Kept each person in one group only, to prevent data leakage.", "Done"),
        ("Export", "Saved the clean table and the statistics.", "Partly done"),
        ("Anonymise, fill gaps, prepare images, package for training",
         "Planned steps. The images are not resized or transformed yet.", "Planned"),
    ], [4.2, 9.7, 2.3])
    g.h2("Why split by patient?")
    g.p("Imagine students who see the exam questions in advance. Their marks look great but "
        "prove nothing. Patients often have several X-rays. If one person's images were in "
        "both the study group and the exam group, the model could recognise the person rather "
        "than the illness. The check found 0 patients in more than one group.")
    g.table(["Group", "Purpose", "Everyday comparison"], [
        ("Train", "The model learns from it.", "The textbook"),
        ("Validation", "Used to adjust the model during work.", "Practice tests"),
        ("Test", "Used once at the end, for the honest score.", "The final exam"),
    ], [3.2, 7.5, 5.5])

    g.h1("7. The plan after this")
    g.bullets([
        "Image branch: a neural network, a type of model that learns visual patterns, reads the X-ray.",
        "Patient branch: a second part reads age, sex and similar details.",
        "Fusion: the two are combined to reach one answer, much as a doctor considers both the "
        "picture and the patient's file.",
        "Explanation: the model should show which image area drove its answer, so a doctor can check it.",
    ])

    g.h1("8. Limits to keep in mind")
    g.bullets([
        "Some labels are wrong, as they were extracted automatically from reports.",
        "Rare conditions have few examples, so results for them will be weaker.",
        "Patients are not evenly represented by age or sex, so a model may work better for some groups.",
        "Medical data is sensitive. Only use data you are allowed to use.",
        "Good test results would not make a tool ready for hospitals. That needs clinical checks and approval.",
    ])

    g.h1("9. Word list")
    g.table(["Word", "Meaning"], WORDS, [4.2, 12])

    g.doc.save(ROOT / "docs" / "explanation-simple.docx")
    print("[explanation] wrote docs/explanation-simple.docx")


if __name__ == "__main__":
    build()
