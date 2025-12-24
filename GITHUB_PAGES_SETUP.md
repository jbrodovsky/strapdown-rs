# GitHub Pages Setup Guide for Strapdown-rs

This guide provides step-by-step instructions for setting up and deploying the Strapdown-rs documentation website using GitHub Pages and mdBook.

## Overview

The strapdown-rs project now includes a comprehensive documentation website built with mdBook. This guide covers:

1. Local development and testing
2. GitHub Pages configuration
3. Automatic deployment via GitHub Actions
4. Custom domain setup

## Project Structure

```
book/
├── book.toml           # mdBook configuration
├── custom.css          # Custom styling
├── src/               # Documentation source files
│   ├── SUMMARY.md     # Table of contents
│   ├── introduction.md
│   ├── quick-start.md
│   ├── installation/
│   ├── user-guide/
│   ├── filters/
│   ├── geonav/
│   ├── gnss/
│   ├── api/
│   ├── examples/
│   ├── development/
│   ├── resources/
│   └── faq.md
└── book/              # Generated HTML (gitignored)
```

## Local Development

### Prerequisites

1. **Install Rust and Cargo** (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Install mdBook**:
   ```bash
   cargo install mdbook
   ```

### Building and Serving Locally

1. **Build the book**:
   ```bash
   cd book
   mdbook build
   ```
   The generated HTML will be in `book/book/`.

2. **Serve the book locally** (with live reload):
   ```bash
   cd book
   mdbook serve
   ```
   Open http://localhost:3000 in your browser.

3. **Watch for changes** (automatic rebuild):
   ```bash
   cd book
   mdbook watch
   ```

### Adding New Content

1. **Create a new markdown file** in the appropriate directory under `book/src/`.

2. **Add the file to SUMMARY.md**:
   ```markdown
   # Summary
   
   - [New Chapter](./path/to/new-chapter.md)
   ```

3. **Build and test** locally before committing.

## GitHub Pages Configuration

### Step 1: Enable GitHub Pages

1. Go to your repository on GitHub: https://github.com/jbrodovsky/strapdown-rs

2. Click **Settings** → **Pages**

3. Under "Build and deployment":
   - **Source**: Select "GitHub Actions"
   - This enables the workflow-based deployment

4. Save the settings.

### Step 2: Verify Workflow Permissions

1. Go to **Settings** → **Actions** → **General**

2. Under "Workflow permissions":
   - Select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"

3. Save the settings.

### Step 3: Deploy

The GitHub Actions workflow (`.github/workflows/deploy-book.yml`) automatically:
- Builds the book when you push to `main`
- Deploys it to GitHub Pages
- Makes it available at: https://jbrodovsky.github.io/strapdown-rs/

**First deployment**: After merging the PR, the site should be live within 2-5 minutes.

## Custom Domain Setup

To use your custom domain (www.strapdown.rs):

### Step 1: Configure DNS

Add these DNS records with your domain registrar:

**For apex domain (strapdown.rs)**:
```
Type: A
Name: @
Value: 185.199.108.153
       185.199.109.153
       185.199.110.153
       185.199.111.153
```

**For www subdomain (www.strapdown.rs)**:
```
Type: CNAME
Name: www
Value: jbrodovsky.github.io
```

### Step 2: Configure GitHub Pages

1. Go to **Settings** → **Pages**

2. Under "Custom domain":
   - Enter: `www.strapdown.rs`
   - Click **Save**

3. Check **Enforce HTTPS** (wait a few minutes for the certificate)

### Step 3: Update book.toml

The `book.toml` already includes:
```toml
[output.html]
cname = "www.strapdown.rs"
site-url = "/"
```

This creates a `CNAME` file in the build output.

### Step 4: Verify

After DNS propagation (can take up to 48 hours but usually minutes):
- Visit https://www.strapdown.rs
- Verify HTTPS works
- Check that redirects work correctly

## Workflow Details

The GitHub Actions workflow (`.github/workflows/deploy-book.yml`) performs these steps:

1. **Checkout**: Gets the repository code
2. **Setup Rust**: Installs the Rust toolchain
3. **Install mdBook**: Installs mdBook via cargo
4. **Build**: Runs `mdbook build` in the `book/` directory
5. **Upload**: Uploads the build artifact
6. **Deploy**: Deploys to GitHub Pages (only on `main` branch)

### Triggering Deployment

Deployment happens automatically when:
- You push to the `main` branch
- A PR is merged into `main`
- You manually trigger via "Run workflow" in the Actions tab

### Monitoring Deployment

1. Go to **Actions** tab in GitHub
2. Click on the latest "Deploy mdBook to GitHub Pages" workflow
3. Check the build and deploy steps
4. If successful, visit the site URL

## Troubleshooting

### Build Fails Locally

**Error**: "command not found: mdbook"
- **Solution**: Install mdBook: `cargo install mdbook`

**Error**: "Invalid configuration file"
- **Solution**: Check `book.toml` syntax with `mdbook build --verbose`

### Build Fails on GitHub

**Error**: Permission denied
- **Solution**: Check workflow permissions in Settings → Actions

**Error**: mdBook installation fails
- **Solution**: The workflow should install mdBook automatically. Check the workflow logs.

### Site Not Updating

1. **Check workflow status**: Actions tab should show successful deployment
2. **Clear browser cache**: Use Ctrl+Shift+R or Cmd+Shift+R
3. **Wait for propagation**: Can take 1-2 minutes after deployment
4. **Check GitHub Pages settings**: Ensure source is set to "GitHub Actions"

### Custom Domain Not Working

1. **Verify DNS records**: Use `dig www.strapdown.rs` or https://dnschecker.org
2. **Check CNAME file**: Should be present in the deployed site
3. **Wait for DNS propagation**: Can take up to 48 hours
4. **Check HTTPS**: May need time to provision certificate

## Customization

### Styling

Edit `book/custom.css` to customize the appearance:
- Colors and themes
- Typography
- Layout and spacing
- Mobile responsiveness

### Configuration

Edit `book/book.toml` to change:
- Site title and description
- Theme preferences
- Search settings
- MathJax support
- Git integration

### Theme

mdBook includes several built-in themes:
- Light (default)
- Navy
- Coal
- Rust
- Ayu

Users can switch themes using the theme selector in the top menu.

## Maintenance

### Updating Content

1. Edit markdown files in `book/src/`
2. Test locally with `mdbook serve`
3. Commit and push to `main`
4. GitHub Actions automatically deploys

### Updating mdBook

To update to the latest mdBook version:

```bash
cargo install mdbook --force
```

Update the version in `.github/workflows/deploy-book.yml` if needed.

### Monitoring

- Check GitHub Actions for build/deployment status
- Monitor GitHub Pages analytics (if enabled)
- Review issues/feedback from users

## Best Practices

1. **Always test locally** before pushing to main
2. **Use descriptive commit messages** for documentation changes
3. **Keep navigation organized** in SUMMARY.md
4. **Cross-reference pages** using relative links
5. **Include code examples** where appropriate
6. **Update the table of contents** when adding pages
7. **Use proper markdown formatting** for consistency

## Additional Resources

- [mdBook Documentation](https://rust-lang.github.io/mdBook/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Markdown Guide](https://www.markdownguide.org/)

## Support

For issues or questions:
- Open an issue on [GitHub](https://github.com/jbrodovsky/strapdown-rs/issues)
- Contact the maintainer: jbrodovsky@temple.edu
