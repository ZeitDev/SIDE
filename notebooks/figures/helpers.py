import os
import shutil
import subprocess
import plotly.express as px
import cairosvg

def save_figure(fig, height=400, name='test', lrtb_margin=(0, 0, 0, 0), standoff=5, fallback=True, folder='methods', skip_sync=False):
    font_size = 15.5
    width = 600
    family = 'Latin Modern Roman, Computer Modern Roman, serif'
    
    # 1. Primary local output path
    base_path = f'notebooks/output/{folder}/'
    os.makedirs(base_path, exist_ok=True)
    
    axis_kwargs = dict(title_font=dict(size=font_size+2, family=family), tickfont=dict(size=font_size, family=family))
    if standoff is not None:
        axis_kwargs['title_standoff'] = standoff
        
    fig.update_xaxes(**axis_kwargs)
    fig.update_yaxes(**axis_kwargs)
    fig.update_annotations(font=dict(size=font_size+2, family=family)) # Subplot titles
    
    # Standardize Colorbars if they exist
    fig.update_coloraxes(
        colorbar_title_font_size=font_size+2,
        colorbar_title_font_family=family,
        colorbar_tickfont_size=font_size,
        colorbar_tickfont_family=family
    )
    
    for trace in fig.data:
        if 'colorbar' in trace:
            trace.update(
                colorbar_title_font_size=font_size+2,
                colorbar_title_font_family=family,
                colorbar_tickfont_size=font_size,
                colorbar_tickfont_family=family
            )

    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=lrtb_margin[0], r=lrtb_margin[1], t=lrtb_margin[2], b=lrtb_margin[3]),
        font=dict(
            family=family,
            size=font_size,
            color='black'
        ),
        legend=dict(
            title_font=dict(family=family, size=font_size+2),
            font=dict(family=family, size=font_size)
        )
    )
    
    fig.show(config={'toImageButtonOptions': {'format': 'svg', 'filename': f'{base_path}{name}'}})
    
    # 2. Save the file locally
    generated_file_path = ""
    if not fallback:
        svg_bytes = fig.to_image(format='svg', width=width, height=height)
        generated_file_path = os.path.join(base_path, f'{name}.pdf')
        cairosvg.svg2pdf(bytestring=svg_bytes, write_to=generated_file_path)
    else:
        generated_file_path = os.path.join(base_path, f'{name}.png')
        fig.write_image(generated_file_path, width=width, height=height, scale=3)
        
    # 3. Automatically sync to Overleaf if a repo path is provided
    if not skip_sync: sync_to_overleaf(generated_file_path, folder, name)


def sync_to_overleaf(source_file, folder, name):
    """Copies the generated image to a local Overleaf Git repo and pushes it."""
    # Define and create target directory inside the Overleaf repository
    repo_path = '/data/Zeitler/masterthesis'
    target_dir = os.path.join(repo_path, 'figures', folder)
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy file over
    filename = os.path.basename(source_file)
    target_file = os.path.join(target_dir, filename)
    shutil.copy2(source_file, target_file)
    
    #subprocess.run(['git', 'add', target_file], cwd=repo_path, check=True, capture_output=True)