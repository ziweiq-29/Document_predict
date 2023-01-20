



import cv2
import layoutparser as lp
image = cv2.imread('test2.jpg')
image = image[..., ::-1]
#load model
model = lp.models.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
                                        )

layout = model.detect(image)

# print("????")

# print layout blocks on the image

text_blocks = lp.Layout([b for b in layout if b.type=='Text'])
figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])

text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

ig1=lp.visualization.draw_box(image, text_blocks,
            box_width=3,
            show_element_id=True)
ig1=ig1.save('geeks2_1.jpg')


ig=lp.visualization.draw_box(image, layout, box_width=3,show_element_type=True,show_element_id=True)

ig = ig.save("geeks2.jpg")
# print("####")

# extract text from layout blocks
ocr_agent = lp.TesseractAgent(languages='eng')

for block in text_blocks:
    segment_image = (block
                       .pad(left=5, right=5, top=5, bottom=5)
                       .crop_image(image))

    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)


for txt in text_blocks.get_texts():
    print(txt, end='\n---\n')