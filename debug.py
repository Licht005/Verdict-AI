from langchain_community.document_loaders import PDFPlumberLoader

loader = PDFPlumberLoader("Doc/Ghana Constitution.pdf")
docs = loader.load()

full_text = " ".join([d.page_content for d in docs])
idx = full_text.find("6")
print(repr(full_text[idx:idx+600]))