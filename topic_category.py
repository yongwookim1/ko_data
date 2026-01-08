import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environment
from collections import Counter
import os

# Set display options to prevent column truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 30)  # Truncate long content at 30 characters
pd.set_option('display.width', 200)

def load_and_compare():
    """Load and compare classification results from different methods."""
    base_path = Path("results_topic")

    # 1. Load data from files
    try:
        with open(base_path / "method1_single.json", "r", encoding="utf-8") as f:
            data_m1 = json.load(f)
        with open(base_path / "method2_multi.json", "r", encoding="utf-8") as f:
            data_m2 = json.load(f)
        with open(base_path / "method3_binary.json", "r", encoding="utf-8") as f:
            data_m3 = json.load(f)
    except FileNotFoundError:
        print("❌ Result files not found. Please run the classification code first.")
        return

    # 2. Organize data into dictionary format
    # Key: Topic, Value: Results from each method
    comparison_data = []

    # Process topics in order
    for i in range(len(data_m1)):
        topic_full = data_m1[i]['topic']

        # Truncate long topics for better display
        topic_short = topic_full[:20] + "..." if len(topic_full) > 20 else topic_full

        # Method 1 result (String)
        res_m1 = data_m1[i]['result']

        # Method 2 result (List -> String)
        res_m2 = ", ".join(data_m2[i]['result'])

        # Method 3 result (List -> String)
        # Show '-' if matched_list is empty (safe case)
        m3_list = data_m3[i].get('matched_list', [])
        res_m3 = ", ".join(m3_list) if m3_list else "-"

        comparison_data.append({
            "Topic": topic_short,
            "Method 1 (single selection)": res_m1,
            "Method 2 (multi seletion)": res_m2,
            "Method 3 (binary selection)": res_m3
        })

    # 3. Create DataFrame and display
    df = pd.DataFrame(comparison_data)

    print("\n" + "="*100)
    print(" 🧐 AI Risk Classification Comparison Table")
    print("="*100)
    print(df.to_string(index=False))  # Clean output without index numbers
    print("="*100)

    # 4. Save as CSV (for Excel viewing)
    csv_path = "comparison_table.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # utf-8-sig prevents Korean character corruption in Excel
    print(f"\n✅ Also saved as CSV file for Excel: {csv_path}")

    # 5. Generate HTML file (with charts)
    html_path = "comparison_table.html"
    chart_files = create_category_distribution_charts(csv_path)
    create_html_table_with_charts(csv_path, html_path, chart_files)
    print(f"✅ Also saved as beautiful HTML file: {html_path}")

def create_html_table(csv_path, html_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    all_categories = set()
    for col in df.columns[1:]:
        for cell in df[col]:
            if pd.notna(cell) and cell != '-':
                categories = [cat.strip() for cat in str(cell).split(',')]
                all_categories.update(categories)

    sorted_categories = sorted(all_categories)
    category_mapping = {cat: f"{i+1}: {cat}" for i, cat in enumerate(sorted_categories)}

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Risk Classification Comparison Table</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        h1 {{
            text-align: center;
            color: #333;
            padding: 20px;
            margin: 0;
            background-color: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        .table-container {{
            overflow-x: auto;
            padding: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0 auto;
            font-size: 14px;
        }}
        th {{
            background-color: #007bff;
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #dee2e6;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        td {{
            padding: 8px;
            border: 1px solid #dee2e6;
            vertical-align: top;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e3f2fd;
        }}
        .topic-cell {{
            max-width: 200px;
            word-wrap: break-word;
            font-weight: 500;
        }}
        .category-cell {{
            max-width: 150px;
        }}
        .category-tag {{
            display: inline-block;
            background-color: #e9ecef;
            color: #495057;
            padding: 2px 6px;
            margin: 1px;
            border-radius: 3px;
            font-size: 12px;
            border: 1px solid #ced4da;
        }}
        .category-list {{
            line-height: 1.4;
        }}
        .legend {{
            background-color: #f8f9fa;
            padding: 20px;
            border-top: 1px solid #dee2e6;
            margin-top: 20px;
        }}
        .legend h3 {{
            margin-top: 0;
            color: #333;
        }}
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .legend-item {{
            background-color: white;
            padding: 8px 12px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            font-size: 13px;
        }}
        .stats {{
            text-align: center;
            padding: 15px;
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
            color: #666;
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            .table-container {{
                padding: 10px;
            }}
            th, td {{
                padding: 6px 4px;
                font-size: 12px;
            }}
            .topic-cell {{
                max-width: 150px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧐 AI Risk Classification Comparison Table<br><small>Classification Results Comparison</small></h1>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Topic (주제)</th>
                        <th>Method 1<br><small>single selection</small></th>
                        <th>Method 2<br><small>multi selection</small></th>
                        <th>Method 3<br><small>binary</small></th>
                    </tr>
                </thead>
                <tbody>
"""

    for _, row in df.iterrows():
        html_content += "                    <tr>\n"
        html_content += f"                        <td class=\"topic-cell\">{row['Topic (주제)']}</td>\n"

        for col in df.columns[1:]:
            cell_value = row[col]
            if pd.isna(cell_value) or cell_value == '-':
                html_content += f"                        <td class=\"category-cell\">-</td>\n"
            else:
                categories = [cat.strip() for cat in str(cell_value).split(',')]
                numbered_categories = [category_mapping.get(cat, cat) for cat in categories]
                category_html = "<div class=\"category-list\">" + "<br>".join(numbered_categories) + "</div>"
                html_content += f"                        <td class=\"category-cell\">{category_html}</td>\n"

        html_content += "                    </tr>\n"

    html_content += f"""                </tbody>
            </table>
        </div>

        <div class="legend">
            <h3>Category Legend</h3>
            <div class="legend-grid">
"""

    for cat, numbered_cat in category_mapping.items():
        html_content += f'                <div class="legend-item"><strong>{numbered_cat}</strong></div>\n'

    html_content += f"""            </div>
        </div>

        <div class="stats">
        </div>
    </div>
</body>
</html>"""

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def create_html_table_with_charts(csv_path, html_path, chart_files):
    """Read CSV file and convert to beautiful HTML table (with charts included)."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    all_categories = set()
    for col in df.columns[1:]:
        for cell in df[col]:
            if pd.notna(cell) and cell != '-':
                categories = [cat.strip() for cat in str(cell).split(',')]
                all_categories.update(categories)

    sorted_categories = sorted(all_categories)
    category_mapping = {cat: f"{i+1}: {cat}" for i, cat in enumerate(sorted_categories)}

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Risk Classification Comparison Table</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        h1 {{
            text-align: center;
            color: #333;
            padding: 20px;
            margin: 0;
            background-color: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }}
        h2 {{
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        .table-container {{
            overflow-x: auto;
            padding: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0 auto;
            font-size: 14px;
        }}
        th {{
            background-color: #007bff;
            color: white;
            padding: 12px 8px;
            text-align: center;
            font-weight: 600;
            border: 1px solid #dee2e6;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        td {{
            padding: 8px;
            border: 1px solid #dee2e6;
            vertical-align: top;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e3f2fd;
        }}
        .topic-cell {{
            max-width: 200px;
            word-wrap: break-word;
            font-weight: 500;
        }}
        .category-cell {{
            max-width: 150px;
        }}
        .category-list {{
            line-height: 1.4;
        }}
        .charts-section {{
            padding: 20px;
        }}
        .chart-container {{
            margin-bottom: 40px;
            text-align: center;
        }}
        .chart-image {{
            max-width: 100%;
            height: auto;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-stats {{
            margin-top: 15px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            display: inline-block;
            min-width: 300px;
        }}
        .legend {{
            background-color: #f8f9fa;
            padding: 20px;
            border-top: 1px solid #dee2e6;
            margin-top: 20px;
        }}
        .legend h3 {{
            margin-top: 0;
            color: #333;
        }}
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }}
        .legend-item {{
            background-color: white;
            padding: 8px 12px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
            font-size: 13px;
        }}
        .stats {{
            text-align: center;
            padding: 15px;
            background-color: #f8f9fa;
            border-top: 1px solid #dee2e6;
            color: #666;
        }}
        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}
            .table-container {{
                padding: 10px;
            }}
            th, td {{
                padding: 6px 4px;
                font-size: 12px;
            }}
            .topic-cell {{
                max-width: 150px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧐 AI Risk Classification Comparison Table<br><small>Classification Results Comparison</small></h1>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Topic (주제)</th>
                        <th>Method 1<br><small>single selection</small></th>
                        <th>Method 2<br><small>multi selection</small></th>
                        <th>Method 3<br><small>binary</small></th>
                    </tr>
                </thead>
                <tbody>
"""

    # 테이블 행 생성
    for _, row in df.iterrows():
        html_content += "                    <tr>\n"
        html_content += f"                        <td class=\"topic-cell\">{row['Topic (주제)']}</td>\n"

        for col in df.columns[1:]:
            cell_value = row[col]
            if pd.isna(cell_value) or cell_value == '-':
                html_content += f"                        <td class=\"category-cell\">-</td>\n"
            else:
                categories = [cat.strip() for cat in str(cell_value).split(',')]
                numbered_categories = [category_mapping.get(cat, cat) for cat in categories]
                category_html = "<div class=\"category-list\">" + "<br>".join(numbered_categories) + "</div>"
                html_content += f"                        <td class=\"category-cell\">{category_html}</td>\n"

        html_content += "                    </tr>\n"

    html_content += f"""                </tbody>
            </table>
        </div>

        <div class="charts-section">
            <h2>📊 Category Distribution Analysis</h2>
"""

    # Show the single concatenated chart
    if chart_files:
        chart_file = chart_files[0][1]  # All entries point to the same concatenated file

        html_content += f"""
            <div class="chart-container">
                <h3>Combined Category Distribution</h3>
                <img src="{chart_file}" alt="Category Distribution Comparison" class="chart-image">
                <div class="chart-stats">
                    <p><strong>Chart shows distribution across all three classification methods</strong></p>
                </div>
            </div>
"""

    html_content += f"""
        </div>

        <div class="legend">
            <h3>Category Legend</h3>
            <div class="legend-grid">
"""

    for cat, numbered_cat in category_mapping.items():
        html_content += f'                <div class="legend-item"><strong>{numbered_cat}</strong></div>\n'

    html_content += f"""            </div>
        </div>

        <div class="stats">
            Finised
        </div>
    </div>
</body>
</html>"""

    # HTML 파일 저장
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def analyze_category_distribution(csv_path):
    """Analyze category distribution for each method."""
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    methods = {
        'Method 1 (1개 강제)': 'single_selection',
        'Method 2 (다중 강제)': 'multi_selection',
        'Method 3': 'binary_selection'
    }

    distributions = {}

    for method_name, short_name in methods.items():
        if method_name not in df.columns:
            continue

        all_categories = []
        for cell in df[method_name]:
            if pd.notna(cell) and cell != '-':
                # Split comma-separated categories
                categories = [cat.strip() for cat in str(cell).split(',')]
                all_categories.extend(categories)

        # Count frequency of each category
        category_counts = Counter(all_categories)
        distributions[short_name] = dict(sorted(category_counts.items()))

    return distributions

def create_category_distribution_charts(csv_path):
    """Create a single concatenated category distribution chart for all methods."""
    distributions = analyze_category_distribution(csv_path)

    # Create directory for charts
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)

    if not distributions:
        return []

    # Set font (optional)
    plt.rcParams['font.family'] = 'DejaVu Sans'  # or 'Malgun Gothic' etc.

    # Get all categories for consistent mapping
    all_categories = set()
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    for col in df.columns[1:]:
        for cell in df[col]:
            if pd.notna(cell) and cell != '-':
                cats = [cat.strip() for cat in str(cell).split(',')]
                all_categories.update(cats)

    sorted_cats = sorted(all_categories)
    category_mapping = {cat: f"{i+1}: {cat}" for i, cat in enumerate(sorted_cats)}

    # Create single figure with 3 subplots horizontally
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('AI Risk Classification Category Distribution Comparison', fontsize=16, fontweight='bold')

    chart_files = []
    method_names = list(distributions.keys())

    for i, (method_name, category_data) in enumerate(distributions.items()):
        if not category_data:
            continue

        ax = axes[i]

        # Prepare data
        categories = list(category_data.keys())
        counts = list(category_data.values())
        numbered_categories = [category_mapping.get(cat, cat) for cat in categories]

        # Bar chart
        bars = ax.bar(range(len(categories)), counts, color='skyblue', edgecolor='navy', linewidth=1)
        method_title = method_name.replace("_", " ").title()
        ax.set_title(f'{method_title}', fontsize=14, fontweight='bold')
        ax.set_xlabel('카테고리', fontsize=12)
        ax.set_ylabel('빈도', fontsize=12)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(numbered_categories, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Display values on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save single concatenated chart
    chart_file = "charts/concatenated_distribution.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Return data for all methods with the single chart file
    for method_name, category_data in distributions.items():
        chart_files.append((method_name, chart_file, category_data))

    print(f"✅ Created concatenated distribution chart: {chart_file}")

    return chart_files

def create_html_from_existing_csv():
    """Generate HTML from existing CSV file (with charts included)."""
    csv_path = "comparison_table.csv"
    html_path = "comparison_table.html"

    if not Path(csv_path).exists():
        print(f"❌ {csv_path} file does not exist.")
        return

    # Create charts
    print("📊 Creating category distribution charts...")
    chart_files = create_category_distribution_charts(csv_path)

    # Generate HTML
    create_html_table_with_charts(csv_path, html_path, chart_files)
    print(f"✅ Beautiful HTML file generated: {html_path}")

if __name__ == "__main__":
    # Generate HTML if CSV file exists
    if Path("comparison_table.csv").exists():
        create_html_from_existing_csv()
    else:
        # Run comparison analysis if JSON files exist
        load_and_compare()