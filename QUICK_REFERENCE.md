# Quick Reference - GitHub Pages Deployment

## After Merge: Immediate Next Steps

### 1. Enable GitHub Pages (2 minutes)

1. Go to https://github.com/jbrodovsky/strapdown-rs/settings/pages
2. Under "Build and deployment":
   - **Source**: Select "GitHub Actions"
3. Click **Save**

### 2. Verify Permissions (1 minute)

1. Go to https://github.com/jbrodovsky/strapdown-rs/settings/actions
2. Under "Workflow permissions":
   - Select **"Read and write permissions"**
   - Check **"Allow GitHub Actions to create and approve pull requests"**
3. Click **Save**

### 3. Trigger First Deployment (automatic)

The merge to `main` will automatically trigger the deployment workflow.

Or manually trigger:
1. Go to https://github.com/jbrodovsky/strapdown-rs/actions
2. Click "Deploy mdBook to GitHub Pages"
3. Click "Run workflow" → "Run workflow"

### 4. Verify Deployment (2-5 minutes)

1. Wait for the workflow to complete (green checkmark)
2. Visit: https://jbrodovsky.github.io/strapdown-rs/
3. Verify the site loads correctly

## Custom Domain Setup (Optional)

### DNS Configuration

Add these records to your domain registrar:

**For strapdown.rs:**
```
Type: A
Name: @
Values:
  185.199.108.153
  185.199.109.153
  185.199.110.153
  185.199.111.153
```

**For www.strapdown.rs:**
```
Type: CNAME
Name: www
Value: jbrodovsky.github.io
```

### GitHub Configuration

1. Go to https://github.com/jbrodovsky/strapdown-rs/settings/pages
2. Under "Custom domain":
   - Enter: `www.strapdown.rs`
   - Click **Save**
3. Wait a few minutes, then check **"Enforce HTTPS"**

### Verify

After DNS propagation (up to 48 hours, usually minutes):
- Visit https://www.strapdown.rs
- Verify HTTPS works
- Check redirect from http:// to https://

## Quick Commands

### Build locally:
```bash
cd book
mdbook build
```

### Serve locally (with live reload):
```bash
cd book
mdbook serve
```

### Check for errors:
```bash
cd book
mdbook build --verbose
```

## File Structure

```
book/
├── book.toml           # Configuration
├── custom.css          # Styling
└── src/
    ├── SUMMARY.md      # Table of contents (edit to add pages)
    └── *.md            # Content pages
```

## Common Tasks

### Add a new page:
1. Create `book/src/path/to/page.md`
2. Add to `book/src/SUMMARY.md`:
   ```markdown
   - [Page Title](./path/to/page.md)
   ```
3. Build and test locally
4. Commit and push

### Update existing page:
1. Edit the markdown file
2. Test locally with `mdbook serve`
3. Commit and push
4. Deployment happens automatically

### Change styling:
1. Edit `book/custom.css`
2. Test locally
3. Commit and push

## Monitoring

### Check deployment status:
https://github.com/jbrodovsky/strapdown-rs/actions

### View deployed site:
https://jbrodovsky.github.io/strapdown-rs/

### Check workflow logs:
Actions tab → Latest "Deploy mdBook to GitHub Pages" run

## Troubleshooting

### Build fails:
```bash
cd book
mdbook build --verbose
```
Check error messages, fix markdown syntax

### Site not updating:
1. Check GitHub Actions (should show green checkmark)
2. Clear browser cache (Ctrl+Shift+R)
3. Wait 1-2 minutes for CDN propagation

### Custom domain not working:
1. Verify DNS: `dig www.strapdown.rs`
2. Check CNAME file is present in deployment
3. Wait for DNS propagation (up to 48 hours)

## Support

- **Setup Guide**: See GITHUB_PAGES_SETUP.md for detailed instructions
- **Project Summary**: See PROJECT_SUMMARY.md for implementation details
- **mdBook Docs**: https://rust-lang.github.io/mdBook/
- **GitHub Pages**: https://docs.github.com/en/pages

## Contact

Questions? Open an issue or contact: jbrodovsky@temple.edu
