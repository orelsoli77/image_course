
import streamlit as st
import cv2
import numpy as np
import skimage.io as io



def segment_image_kmeans(img, k=3, attempts=10): 

    # Convert MxNx3 image into Kx3 where K=MxN
    pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case MxN

    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    
   
    centers = np.uint8(centers)

    labels = labels.flatten()
    
    segmented_image = centers[labels.flatten()]
    
    segmented_image = segmented_image.reshape(img.shape)
    
    return segmented_image



DEMO_IMAGE = 'demo.png' # a demo image for the segmentation page, if none is uploaded
favicon = 'favicon.png'

st.set_page_config(page_title='K-Means - Yedidya Harris', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')
st.title('Image Segmentation using K-Means, by Yedidya Harris')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
        width: 350px
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
        width: 350px
        margin-left: -350px
    }    
    </style>
    
    """,
    unsafe_allow_html=True,


)

st.sidebar.title('Segmentation Sidebar')
st.sidebar.subheader('Site Pages')

@st.cache()

def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
        

    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized


app_mode = st.sidebar.selectbox('Navigate',
                                  ['About App', 'Segment an Image'])

if app_mode == 'About App':
    st.markdown('In this app we will segment images using K-Means')
    
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )


    st.video('https://www.youtube.com/watch?v=6CqRnx6Ic48')


    st.markdown('''
                ## About the app \n
                Hey, this web app is a great one to segment images using K-Means. \n
                There are many way. \n
                Enjoy! Yedidya


                ''')


if app_mode == 'Segment an Image':
    
    st.sidebar.markdown('---') # adds a devider (a line)
    
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )

    k_value = st.sidebar.number_input('Insert K value (number of clusters):', value=4, min_value = 1) # asks for input from the user
    st.sidebar.markdown('---') # adds a devider (a line)
    
    attempts_value_slider = st.sidebar.slider('Number of attempts', value = 7, min_value = 1, max_value = 10) # slider example
    st.sidebar.markdown('---') # adds a devider (a line)
    
    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if img_file_buffer is not None:
        image = io.imread(img_file_buffer)
    else: # if no image was uploaded, then segment the demo image
        demo_image = DEMO_IMAGE
        image = io.imread(demo_image)

    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
    segmented_image = segment_image_kmeans(image, k=k_value, attempts=attempts_value_slider)
    
    st.subheader('Output Image')
    st.image(segmented_image, use_column_width=True)


