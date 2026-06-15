# COSMOS AI @ UC Merced

Course modules for the UC Merced COSMOS (California State Summer School for Mathematics and Science) Artificial Intelligence cluster.

**Live site:** https://cosmos-ai-ucm.github.io

## Setup

This site uses [Jekyll](https://jekyllrb.com/) and is hosted on GitHub Pages.

### Local Development

1. Install Ruby and Jekyll
2. Clone this repository
3. Run `bundle install` (if using a Gemfile)
4. Run `jekyll serve`
5. Visit `http://localhost:4000`

### Deployment

Push to the `main` branch and GitHub Pages will automatically build and deploy the site.

## Structure

```
├── _config.yml          # Site configuration
├── _layouts/
│   └── default.html     # Main layout template
├── assets/
│   └── css/
│       └── style.css    # Main stylesheet
├── index.html           # Homepage
└── modules.html         # Module listing
```

## Customization

- Edit `_config.yml` to update site metadata and navigation
- Add module pages and link them from `modules.html`
