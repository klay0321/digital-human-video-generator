import inspect

import streamlit as st


def st_image_compat(image, caption=None, width=None, use_container_width=True, **kwargs):
    """
    Compatible wrapper for st.image across Streamlit versions.
    Newer Streamlit supports use_container_width.
    Older Streamlit supports use_column_width.
    """
    params = inspect.signature(st.image).parameters

    if "use_container_width" in params:
        return st.image(
            image,
            caption=caption,
            width=width,
            use_container_width=use_container_width,
            **kwargs,
        )

    if "use_column_width" in params:
        return st.image(
            image,
            caption=caption,
            width=width,
            use_column_width=use_container_width,
            **kwargs,
        )

    return st.image(
        image,
        caption=caption,
        width=width,
        **kwargs,
    )
