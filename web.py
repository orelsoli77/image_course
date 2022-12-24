{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/orelsoli77/image_course/blob/main/web.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Create your first 'real' web-app, with streamlit (only Python!)\n",
        "\n",
        "Yedidya Harris"
      ],
      "metadata": {
        "id": "jcCDdaiRK37A"
      },
      "id": "jcCDdaiRK37A"
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Docs and Examples:\n",
        "\n",
        "\n",
        "*   Examples: https://streamlit.io/gallery\n",
        "*   My exampls for class: https://bit.ly/yedidyaCV\n",
        "*   Docs: https://docs.streamlit.io/\n",
        "\n"
      ],
      "metadata": {
        "id": "eTCoapCt_6Xk"
      },
      "id": "eTCoapCt_6Xk"
    },
    {
      "cell_type": "markdown",
      "source": [
        "## importing libs and creating functions"
      ],
      "metadata": {
        "id": "bT6Nqvz-K-ax"
      },
      "id": "bT6Nqvz-K-ax"
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kpGQQ3LVfWGj",
        "outputId": "e3bfe441-6f0c-4acc-daf6-d9af341755b3"
      },
      "id": "kpGQQ3LVfWGj",
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting streamlit\n",
            "  Downloading streamlit-1.16.0-py2.py3-none-any.whl (9.2 MB)\n",
            "\u001b[K     |████████████████████████████████| 9.2 MB 4.2 MB/s \n",
            "\u001b[?25hRequirement already satisfied: click>=7.0 in /usr/local/lib/python3.8/dist-packages (from streamlit) (7.1.2)\n",
            "Requirement already satisfied: importlib-metadata>=1.4 in /usr/local/lib/python3.8/dist-packages (from streamlit) (5.1.0)\n",
            "Requirement already satisfied: altair>=3.2.0 in /usr/local/lib/python3.8/dist-packages (from streamlit) (4.2.0)\n",
            "Requirement already satisfied: packaging>=14.1 in /usr/local/lib/python3.8/dist-packages (from streamlit) (21.3)\n",
            "Collecting gitpython!=3.1.19\n",
            "  Downloading GitPython-3.1.29-py3-none-any.whl (182 kB)\n",
            "\u001b[K     |████████████████████████████████| 182 kB 58.9 MB/s \n",
            "\u001b[?25hRequirement already satisfied: cachetools>=4.0 in /usr/local/lib/python3.8/dist-packages (from streamlit) (5.2.0)\n",
            "Requirement already satisfied: toml in /usr/local/lib/python3.8/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions>=3.10.0.0 in /usr/local/lib/python3.8/dist-packages (from streamlit) (4.4.0)\n",
            "Collecting watchdog\n",
            "  Downloading watchdog-2.2.0-py3-none-manylinux2014_x86_64.whl (78 kB)\n",
            "\u001b[K     |████████████████████████████████| 78 kB 8.8 MB/s \n",
            "\u001b[?25hCollecting pydeck>=0.1.dev5\n",
            "  Downloading pydeck-0.8.0-py2.py3-none-any.whl (4.7 MB)\n",
            "\u001b[K     |████████████████████████████████| 4.7 MB 59.4 MB/s \n",
            "\u001b[?25hRequirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from streamlit) (1.21.6)\n",
            "Requirement already satisfied: pyarrow>=4.0 in /usr/local/lib/python3.8/dist-packages (from streamlit) (9.0.0)\n",
            "Collecting validators>=0.2\n",
            "  Downloading validators-0.20.0.tar.gz (30 kB)\n",
            "Collecting blinker>=1.0.0\n",
            "  Downloading blinker-1.5-py2.py3-none-any.whl (12 kB)\n",
            "Collecting semver\n",
            "  Downloading semver-2.13.0-py2.py3-none-any.whl (12 kB)\n",
            "Requirement already satisfied: protobuf<4,>=3.12 in /usr/local/lib/python3.8/dist-packages (from streamlit) (3.19.6)\n",
            "Collecting rich>=10.11.0\n",
            "  Downloading rich-12.6.0-py3-none-any.whl (237 kB)\n",
            "\u001b[K     |████████████████████████████████| 237 kB 51.4 MB/s \n",
            "\u001b[?25hRequirement already satisfied: python-dateutil in /usr/local/lib/python3.8/dist-packages (from streamlit) (2.8.2)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.8/dist-packages (from streamlit) (7.1.2)\n",
            "Requirement already satisfied: pandas>=0.21.0 in /usr/local/lib/python3.8/dist-packages (from streamlit) (1.3.5)\n",
            "Collecting pympler>=0.9\n",
            "  Downloading Pympler-1.0.1-py3-none-any.whl (164 kB)\n",
            "\u001b[K     |████████████████████████████████| 164 kB 73.8 MB/s \n",
            "\u001b[?25hRequirement already satisfied: tzlocal>=1.1 in /usr/local/lib/python3.8/dist-packages (from streamlit) (1.5.1)\n",
            "Requirement already satisfied: requests>=2.4 in /usr/local/lib/python3.8/dist-packages (from streamlit) (2.23.0)\n",
            "Requirement already satisfied: tornado>=5.0 in /usr/local/lib/python3.8/dist-packages (from streamlit) (6.0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.8/dist-packages (from altair>=3.2.0->streamlit) (2.11.3)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.8/dist-packages (from altair>=3.2.0->streamlit) (4.3.3)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.8/dist-packages (from altair>=3.2.0->streamlit) (0.12.0)\n",
            "Requirement already satisfied: entrypoints in /usr/local/lib/python3.8/dist-packages (from altair>=3.2.0->streamlit) (0.4)\n",
            "Collecting gitdb<5,>=4.0.1\n",
            "  Downloading gitdb-4.0.10-py3-none-any.whl (62 kB)\n",
            "\u001b[K     |████████████████████████████████| 62 kB 1.5 MB/s \n",
            "\u001b[?25hCollecting smmap<6,>=3.0.1\n",
            "  Downloading smmap-5.0.0-py3-none-any.whl (24 kB)\n",
            "Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.8/dist-packages (from importlib-metadata>=1.4->streamlit) (3.11.0)\n",
            "Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=3.0->altair>=3.2.0->streamlit) (5.10.1)\n",
            "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=3.0->altair>=3.2.0->streamlit) (0.19.2)\n",
            "Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=3.0->altair>=3.2.0->streamlit) (22.1.0)\n",
            "Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.8/dist-packages (from packaging>=14.1->streamlit) (3.0.9)\n",
            "Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.8/dist-packages (from pandas>=0.21.0->streamlit) (2022.6)\n",
            "Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.8/dist-packages (from jinja2->altair>=3.2.0->streamlit) (2.0.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil->streamlit) (1.15.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests>=2.4->streamlit) (2022.12.7)\n",
            "Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.8/dist-packages (from requests>=2.4->streamlit) (3.0.4)\n",
            "Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests>=2.4->streamlit) (1.24.3)\n",
            "Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests>=2.4->streamlit) (2.10)\n",
            "Collecting commonmark<0.10.0,>=0.9.0\n",
            "  Downloading commonmark-0.9.1-py2.py3-none-any.whl (51 kB)\n",
            "\u001b[K     |████████████████████████████████| 51 kB 7.6 MB/s \n",
            "\u001b[?25hRequirement already satisfied: pygments<3.0.0,>=2.6.0 in /usr/local/lib/python3.8/dist-packages (from rich>=10.11.0->streamlit) (2.6.1)\n",
            "Requirement already satisfied: decorator>=3.4.0 in /usr/local/lib/python3.8/dist-packages (from validators>=0.2->streamlit) (4.4.2)\n",
            "Building wheels for collected packages: validators\n",
            "  Building wheel for validators (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for validators: filename=validators-0.20.0-py3-none-any.whl size=19581 sha256=d74d2dca5fa62d5d74dcf4b2b4638fb333a998e4265dddd3c199e998629a53f1\n",
            "  Stored in directory: /root/.cache/pip/wheels/19/09/72/3eb74d236bb48bd0f3c6c3c83e4e0c5bbfcbcad7c6c3539db8\n",
            "Successfully built validators\n",
            "Installing collected packages: smmap, gitdb, commonmark, watchdog, validators, semver, rich, pympler, pydeck, gitpython, blinker, streamlit\n",
            "Successfully installed blinker-1.5 commonmark-0.9.1 gitdb-4.0.10 gitpython-3.1.29 pydeck-0.8.0 pympler-1.0.1 rich-12.6.0 semver-2.13.0 smmap-5.0.0 streamlit-1.16.0 validators-0.20.0 watchdog-2.2.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "id": "3d733a8e",
      "metadata": {
        "lines_to_next_cell": 2,
        "id": "3d733a8e"
      },
      "outputs": [],
      "source": [
        "# import libs\n",
        "import streamlit as st\n",
        "import cv2\n",
        "import numpy as np\n",
        "import skimage.io as io\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "# check versions\n",
        "#np.__version__"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "numpy.__version__"
      ],
      "metadata": {
        "id": "j2wG0jN8Vzjl",
        "outputId": "d15cd1de-f2ee-4461-edb3-f27d32783b8b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        }
      },
      "id": "j2wG0jN8Vzjl",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'1.21.6'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "9c37e0a4",
      "metadata": {
        "id": "9c37e0a4"
      },
      "outputs": [],
      "source": [
        "# function to segment using k-means\n",
        "\n",
        "def segment_image_kmeans(img, k=3, attempts=10): \n",
        "\n",
        "    # Convert MxNx3 image into Kx3 where K=MxN\n",
        "    pixel_values  = img.reshape((-1,3))  #-1 reshape means, in this case MxN\n",
        "\n",
        "    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV\n",
        "    pixel_values = np.float32(pixel_values)\n",
        "\n",
        "    # define stopping criteria\n",
        "    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)\n",
        "    \n",
        "    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)\n",
        "    \n",
        "    # convert back to 8 bit values\n",
        "    centers = np.uint8(centers)\n",
        "\n",
        "    # flatten the labels array\n",
        "    labels = labels.flatten()\n",
        "    \n",
        "    # convert all pixels to the color of the centroids\n",
        "    segmented_image = centers[labels.flatten()]\n",
        "    \n",
        "    # reshape back to the original image dimension\n",
        "    segmented_image = segmented_image.reshape(img.shape)\n",
        "    \n",
        "    return segmented_image"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## vars, main page, and sidebar"
      ],
      "metadata": {
        "id": "VbFKJzdhLEg7"
      },
      "id": "VbFKJzdhLEg7"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "735964f5",
      "metadata": {
        "lines_to_next_cell": 2,
        "id": "735964f5"
      },
      "outputs": [],
      "source": [
        "# vars\n",
        "DEMO_IMAGE = 'demo.png' # a demo image for the segmentation page, if none is uploaded\n",
        "favicon = 'favicon.png'\n",
        "\n",
        "# main page\n",
        "st.set_page_config(page_title='K-Means - Yedidya Harris', page_icon = favicon, layout = 'wide', initial_sidebar_state = 'auto')\n",
        "st.title('Image Segmentation using K-Means, by Yedidya Harris')\n",
        "\n",
        "# side bar\n",
        "st.markdown(\n",
        "    \"\"\"\n",
        "    <style>\n",
        "    [data-testid=\"stSidebar\"][aria-expanded=\"true\"] . div:first-child{\n",
        "        width: 350px\n",
        "    }\n",
        "    \n",
        "    [data-testid=\"stSidebar\"][aria-expanded=\"false\"] . div:first-child{\n",
        "        width: 350px\n",
        "        margin-left: -350px\n",
        "    }    \n",
        "    </style>\n",
        "    \n",
        "    \"\"\",\n",
        "    unsafe_allow_html=True,\n",
        "\n",
        "\n",
        ")\n",
        "\n",
        "st.sidebar.title('Segmentation Sidebar')\n",
        "st.sidebar.subheader('Site Pages')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "967ee042",
      "metadata": {
        "lines_to_next_cell": 2,
        "id": "967ee042"
      },
      "outputs": [],
      "source": [
        "# using st.cache so streamlit runs the following function only once, and stores in chache (until changed)\n",
        "@st.cache()\n",
        "\n",
        "# take an image, and return a resized that fits our page\n",
        "def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):\n",
        "    dim = None\n",
        "    (h,w) = image.shape[:2]\n",
        "    \n",
        "    if width is None and height is None:\n",
        "        return image\n",
        "    \n",
        "    if width is None:\n",
        "        r = width/float(w)\n",
        "        dim = (int(w*r),height)\n",
        "    \n",
        "    else:\n",
        "        r = width/float(w)\n",
        "        dim = (width, int(h*r))\n",
        "        \n",
        "    # resize the image\n",
        "    resized = cv2.resize(image, dim, interpolation=inter)\n",
        "    \n",
        "    return resized"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## dropdown menu on the sidebar, to navigate between pages"
      ],
      "metadata": {
        "id": "2wlbwhM2LpG-"
      },
      "id": "2wlbwhM2LpG-"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "d8e73676",
      "metadata": {
        "lines_to_next_cell": 2,
        "id": "d8e73676"
      },
      "outputs": [],
      "source": [
        "# add dropdown to select pages on left\n",
        "app_mode = st.sidebar.selectbox('Navigate',\n",
        "                                  ['About App', 'Segment an Image'])"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## About page"
      ],
      "metadata": {
        "id": "L5Z1_v8_LtCw"
      },
      "id": "L5Z1_v8_LtCw"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "0490187c",
      "metadata": {
        "lines_to_next_cell": 2,
        "id": "0490187c"
      },
      "outputs": [],
      "source": [
        "# About page\n",
        "if app_mode == 'About App':\n",
        "    st.markdown('In this app we will segment images using K-Means')\n",
        "    \n",
        "    \n",
        "    # side bar\n",
        "    st.markdown(\n",
        "        \"\"\"\n",
        "        <style>\n",
        "        [data-testid=\"stSidebar\"][aria-expanded=\"true\"] . div:first-child{\n",
        "            width: 350px\n",
        "        }\n",
        "\n",
        "        [data-testid=\"stSidebar\"][aria-expanded=\"false\"] . div:first-child{\n",
        "            width: 350px\n",
        "            margin-left: -350px\n",
        "        }    \n",
        "        </style>\n",
        "\n",
        "        \"\"\",\n",
        "        unsafe_allow_html=True,\n",
        "\n",
        "\n",
        "    )\n",
        "\n",
        "    # add a video to the page\n",
        "    st.video('https://www.youtube.com/watch?v=6CqRnx6Ic48')\n",
        "\n",
        "\n",
        "    st.markdown('''\n",
        "                ## About the app \\n\n",
        "                Hey, this web app is a great one to segment images using K-Means. \\n\n",
        "                There are many way. \\n\n",
        "                Enjoy! Yedidya\n",
        "\n",
        "\n",
        "                ''') "
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## 'Segment an Image' page"
      ],
      "metadata": {
        "id": "1tZA2YRLLwZS"
      },
      "id": "1tZA2YRLLwZS"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "6f539e86",
      "metadata": {
        "lines_to_next_cell": 3,
        "id": "6f539e86"
      },
      "outputs": [],
      "source": [
        "# Run image\n",
        "if app_mode == 'Segment an Image':\n",
        "    \n",
        "    st.sidebar.markdown('---') # adds a devider (a line)\n",
        "    \n",
        "    # side bar\n",
        "    st.markdown(\n",
        "        \"\"\"\n",
        "        <style>\n",
        "        [data-testid=\"stSidebar\"][aria-expanded=\"true\"] . div:first-child{\n",
        "            width: 350px\n",
        "        }\n",
        "\n",
        "        [data-testid=\"stSidebar\"][aria-expanded=\"false\"] . div:first-child{\n",
        "            width: 350px\n",
        "            margin-left: -350px\n",
        "        }    \n",
        "        </style>\n",
        "\n",
        "        \"\"\",\n",
        "        unsafe_allow_html=True,\n",
        "\n",
        "\n",
        "    )\n",
        "\n",
        "    # choosing a k value (either with +- or with a slider)\n",
        "    k_value = st.sidebar.number_input('Insert K value (number of clusters):', value=4, min_value = 1) # asks for input from the user\n",
        "    st.sidebar.markdown('---') # adds a devider (a line)\n",
        "    \n",
        "    attempts_value_slider = st.sidebar.slider('Number of attempts', value = 7, min_value = 1, max_value = 10) # slider example\n",
        "    st.sidebar.markdown('---') # adds a devider (a line)\n",
        "    \n",
        "    # read an image from the user\n",
        "    img_file_buffer = st.sidebar.file_uploader(\"Upload an image\", type=['jpg', 'jpeg', 'png'])\n",
        "\n",
        "    # assign the uplodaed image from the buffer, by reading it in\n",
        "    if img_file_buffer is not None:\n",
        "        image = io.imread(img_file_buffer)\n",
        "    else: # if no image was uploaded, then segment the demo image\n",
        "        demo_image = DEMO_IMAGE\n",
        "        image = io.imread(demo_image)\n",
        "\n",
        "    # display on the sidebar the uploaded image\n",
        "    st.sidebar.text('Original Image')\n",
        "    st.sidebar.image(image)\n",
        "    \n",
        "    # call the function to segment the image\n",
        "    segmented_image = segment_image_kmeans(image, k=k_value, attempts=attempts_value_slider)\n",
        "    \n",
        "    # Display the result on the right (main frame)\n",
        "    st.subheader('Output Image')\n",
        "    st.image(segmented_image, use_column_width=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Deploy to GitHub + Streamlit Cloud\n"
      ],
      "metadata": {
        "id": "cX3qYMliAIt4"
      },
      "id": "cX3qYMliAIt4"
    },
    {
      "cell_type": "markdown",
      "source": [
        "### ipynb to py"
      ],
      "metadata": {
        "id": "T2to8MzSJKco"
      },
      "id": "T2to8MzSJKco"
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "d4oBGNJi8rhE",
        "outputId": "fafb9187-bf41-46ad-f9a8-80b9ca7874fc"
      },
      "id": "d4oBGNJi8rhE",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# first remove the markdown cells (because it causes problems in streamlit app)\n",
        "\n",
        "import nbformat as nbf\n",
        "folder_path =r'/content/drive/MyDrive/71254_2023/01_Lectures/Class07/kmeans_webapp_yHarris'\n",
        "ntbk_name_to_convert = 'kmeans.ipynb' # find it in your drive, or upload it to the content\n",
        "ntbk = nbf.read(f'{folder_path}/kmeans.ipynb', nbf.NO_CONVERT)\n",
        "new_ntbk = ntbk\n",
        "new_ntbk.cells = [cell for cell in ntbk.cells if cell.cell_type != \"markdown\"]\n",
        "nbf.write(new_ntbk, f'new_{ntbk_name_to_convert}', version=nbf.NO_CONVERT)"
      ],
      "metadata": {
        "id": "vSXLa5cD-qxH"
      },
      "id": "vSXLa5cD-qxH",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# then convert to .py format\n",
        "!pip install ipython\n",
        "!pip install nbconvert\n",
        "\n",
        "# the conversion (it'll save the .py file on the left, in the content)\n",
        "!ipython nbconvert new_kmeans.ipynb --to python"
      ],
      "metadata": {
        "id": "QMit_Ozq_LCQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "c375e538-868e-4950-aa7a-b59bdf9988d4"
      },
      "id": "QMit_Ozq_LCQ",
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Requirement already satisfied: ipython in /usr/local/lib/python3.8/dist-packages (7.9.0)\n",
            "Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.8/dist-packages (from ipython) (5.6.0)\n",
            "Requirement already satisfied: pexpect in /usr/local/lib/python3.8/dist-packages (from ipython) (4.8.0)\n",
            "Requirement already satisfied: decorator in /usr/local/lib/python3.8/dist-packages (from ipython) (4.4.2)\n",
            "Requirement already satisfied: prompt-toolkit<2.1.0,>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from ipython) (2.0.10)\n",
            "Requirement already satisfied: jedi>=0.10 in /usr/local/lib/python3.8/dist-packages (from ipython) (0.18.2)\n",
            "Requirement already satisfied: pygments in /usr/local/lib/python3.8/dist-packages (from ipython) (2.6.1)\n",
            "Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.8/dist-packages (from ipython) (57.4.0)\n",
            "Requirement already satisfied: pickleshare in /usr/local/lib/python3.8/dist-packages (from ipython) (0.7.5)\n",
            "Requirement already satisfied: backcall in /usr/local/lib/python3.8/dist-packages (from ipython) (0.2.0)\n",
            "Requirement already satisfied: parso<0.9.0,>=0.8.0 in /usr/local/lib/python3.8/dist-packages (from jedi>=0.10->ipython) (0.8.3)\n",
            "Requirement already satisfied: wcwidth in /usr/local/lib/python3.8/dist-packages (from prompt-toolkit<2.1.0,>=2.0.0->ipython) (0.2.5)\n",
            "Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.8/dist-packages (from prompt-toolkit<2.1.0,>=2.0.0->ipython) (1.15.0)\n",
            "Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.8/dist-packages (from pexpect->ipython) (0.7.0)\n",
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Requirement already satisfied: nbconvert in /usr/local/lib/python3.8/dist-packages (5.6.1)\n",
            "Requirement already satisfied: testpath in /usr/local/lib/python3.8/dist-packages (from nbconvert) (0.6.0)\n",
            "Requirement already satisfied: jupyter-core in /usr/local/lib/python3.8/dist-packages (from nbconvert) (5.1.0)\n",
            "Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (5.6.0)\n",
            "Requirement already satisfied: pygments in /usr/local/lib/python3.8/dist-packages (from nbconvert) (2.6.1)\n",
            "Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (0.4)\n",
            "Requirement already satisfied: bleach in /usr/local/lib/python3.8/dist-packages (from nbconvert) (5.0.1)\n",
            "Requirement already satisfied: defusedxml in /usr/local/lib/python3.8/dist-packages (from nbconvert) (0.7.1)\n",
            "Requirement already satisfied: jinja2>=2.4 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (2.11.3)\n",
            "Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (1.5.0)\n",
            "Requirement already satisfied: nbformat>=4.4 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (5.7.0)\n",
            "Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.8/dist-packages (from nbconvert) (0.8.4)\n",
            "Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.8/dist-packages (from jinja2>=2.4->nbconvert) (2.0.1)\n",
            "Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.8/dist-packages (from nbformat>=4.4->nbconvert) (4.3.3)\n",
            "Requirement already satisfied: fastjsonschema in /usr/local/lib/python3.8/dist-packages (from nbformat>=4.4->nbconvert) (2.16.2)\n",
            "Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (22.1.0)\n",
            "Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (5.10.0)\n",
            "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat>=4.4->nbconvert) (0.19.2)\n",
            "Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.8/dist-packages (from importlib-resources>=1.4.0->jsonschema>=2.6->nbformat>=4.4->nbconvert) (3.11.0)\n",
            "Requirement already satisfied: webencodings in /usr/local/lib/python3.8/dist-packages (from bleach->nbconvert) (0.5.1)\n",
            "Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.8/dist-packages (from bleach->nbconvert) (1.15.0)\n",
            "Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.8/dist-packages (from jupyter-core->nbconvert) (2.5.4)\n",
            "[TerminalIPythonApp] WARNING | Subcommand `ipython nbconvert` is deprecated and will be removed in future versions.\n",
            "[TerminalIPythonApp] WARNING | You likely want to use `jupyter nbconvert` in the future\n",
            "[NbConvertApp] Converting notebook new_kmeans.ipynb to python\n",
            "[NbConvertApp] Writing 6662 bytes to new_kmeans.py\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Create a requirements text file"
      ],
      "metadata": {
        "id": "5apyfPSWIHxY"
      },
      "id": "5apyfPSWIHxY"
    },
    {
      "cell_type": "code",
      "source": [
        "# add more libraries if you used! as a new line\n",
        "with open('requirements.txt', 'w') as f:\n",
        "    f.write('''streamlit\n",
        "scikit-image\n",
        "opencv-contrib-python-headless\n",
        "numpy''')"
      ],
      "metadata": {
        "id": "j5tQux9bIJ9z"
      },
      "id": "j5tQux9bIJ9z",
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Upload the app.py and the requirements.txt file to GitHub manually (in the future we can learn how to use git)\n"
      ],
      "metadata": {
        "id": "F_3C3rEZA9oh"
      },
      "id": "F_3C3rEZA9oh"
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "1.   Open a GitHub repo, call it 'apps'.\n",
        "2.   In it, create another folder with the name of your app.\n",
        "3.   Upload to that folder the .py + .txt files.\n",
        "\n"
      ],
      "metadata": {
        "id": "pK0jn550-U-W"
      },
      "id": "pK0jn550-U-W"
    },
    {
      "cell_type": "markdown",
      "source": [
        "### Setup a new app on Streamlit cloud"
      ],
      "metadata": {
        "id": "z4c04kfK-r6m"
      },
      "id": "z4c04kfK-r6m"
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "1.   First signup here, using your GitHub acc: https://share.streamlit.io/signup\n",
        "2.   Then create a new app here: https://share.streamlit.io \n",
        "3.   Choose your GitHub repo, the branch (usually 'main'), and the app.py file path.\n",
        "3.   Click on deploy.\n",
        "4.   Create a shorturl for your app with tinyurl.com or bit.ly\n",
        "\n"
      ],
      "metadata": {
        "id": "92r8MgzT-z7M"
      },
      "id": "92r8MgzT-z7M"
    }
  ],
  "metadata": {
    "jupytext": {
      "cell_metadata_filter": "-all",
      "encoding": "# coding: utf-8",
      "executable": "/usr/bin/env python",
      "main_language": "python",
      "notebook_metadata_filter": "-all"
    },
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "bT6Nqvz-K-ax"
      ],
      "include_colab_link": true
    },
    "language_info": {
      "name": "python"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "gpuClass": "standard"
  },
  "nbformat": 4,
  "nbformat_minor": 5
}