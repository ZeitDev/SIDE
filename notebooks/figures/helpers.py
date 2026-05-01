import os
import plotly.express as px
import cairosvg

def save_figure(fig, height=400, name='test', lrtb_margin=(0, 0, 0, 0), standoff=5, fallback=True):
    font_size = 15.5
    width = 600
    family= 'Latin Modern Roman, Computer Modern Roman, serif'
    base_path = 'notebooks/output/methods/'
    
    fig.update_xaxes(title_standoff=standoff, title_font=dict(size=font_size, family=family))
    fig.update_yaxes(title_standoff=standoff, title_font=dict(size=font_size, family=family))

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=lrtb_margin[0], r=lrtb_margin[1], t=lrtb_margin[2], b=lrtb_margin[3]),
        font=dict(
            family=family,
            size=font_size,
            color='black'
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
            scale=3
        )
    