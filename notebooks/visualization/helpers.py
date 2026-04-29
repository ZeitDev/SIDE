import os
import plotly.express as px
import cairosvg

def save_figure(fig, height=400, name='test', margin=(0, 0, 0, 0), standoff=5, fallback=True):
    font_size = 11
    font_size *= 1.333
    width = 600
    family= 'Latin Modern Roman, Computer Modern Roman, serif'
    base_path = 'notebooks/output/methods/'
    
    fig.update_xaxes(title_standoff=standoff, title_font=dict(size=font_size, family=family))
    fig.update_yaxes(title_standoff=standoff, title_font=dict(size=font_size, family=family))

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=margin[0], r=margin[1], t=margin[2], b=margin[3]),
        font=dict(
            family=family,
            size=font_size
        )
    )
    
    fig.show(config={'toImageButtonOptions': {'format': 'svg', 'filename': f'notebooks/output/methods/{name}'}})
    if not fallback:
        svg_bytes = fig.to_image(
            format='svg', 
            width=width,
            height=height,
        )
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=os.path.join(base_path, f'{name}.pdf'))
    else:
        fig.write_image(
            os.path.join(base_path, f'{name}.png'),
            width=width,
            height=height,
            scale=5  # This is the magic number. Increase to 6 or 8 if you need it even larger.
        )
    