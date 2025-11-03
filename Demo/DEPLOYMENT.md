# Deployment Guide for CAR T-Cell Demo on Render

This guide walks you through deploying the Flask-based CAR T-Cell manufacturing demo to Render.

## 📋 Prerequisites

- GitHub/GitLab account
- Render account (free tier works fine)
- Git installed locally

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Initialize git repository** (if not already done):
   ```bash
   cd /home/emms/Downloads/CarTCell/Demo
   git init
   git add .
   git commit -m "Initial commit with deployment files"
   ```

2. **Push to GitHub/GitLab**:
   ```bash
   # Create a new repository on GitHub/GitLab first, then:
   git remote add origin <your-repo-url>
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy to Render

#### Option A: Using Render Dashboard (Recommended for first-time)

1. **Log in to Render**: Go to https://render.com and sign in

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub/GitLab account if not already connected
   - Select your repository

3. **Configure the Service**:
   - **Name**: `car-t-cell-demo` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: Leave empty (or set to `Demo` if deploying from repo root)
   - **Environment**: `Docker`
   - **Plan**: `Free` or `Starter` (Free tier may have cold starts)

4. **Advanced Settings** (optional):
   - Auto-Deploy: `Yes` (recommended)
   - Health Check Path: `/`

5. **Click "Create Web Service"**

#### Option B: Using render.yaml (Infrastructure as Code)

1. In your repository root, the `render.yaml` file is already configured

2. On Render Dashboard:
   - Click "New +" → "Blueprint"
   - Select your repository
   - Render will automatically detect `render.yaml`
   - Click "Apply"

### Step 3: Monitor Deployment

1. **Watch Build Logs**: 
   - Render will show real-time build logs
   - Docker image build takes 3-5 minutes (first time)
   - Subsequent builds use caching and are faster

2. **Check for Success**:
   - Look for "Build successful" message
   - Service will start automatically
   - You'll get a URL like: `https://car-t-cell-demo.onrender.com`

3. **Test Your Application**:
   - Visit the provided URL
   - Test WebSocket connection by running a simulation
   - Try both Standard Protocol and AI Strategy scenarios

## 🔧 Configuration

### Environment Variables

No additional environment variables are required. The application uses:
- `PORT`: Automatically set by Render (default: 10000)
- `PYTHON_VERSION`: Set to 3.11 in render.yaml

### Custom Domain (Optional)

To use a custom domain:
1. Go to your service settings on Render
2. Click "Custom Domain"
3. Follow instructions to add your domain
4. Update DNS records as specified

## 📊 Monitoring

### Health Checks

Render performs automatic health checks on the `/` endpoint:
- Every 30 seconds in production
- Service restarts automatically if unhealthy

### Logs

View real-time logs:
1. Go to your service in Render Dashboard
2. Click "Logs" tab
3. Filter by timeframe or search for specific events

## 🐛 Troubleshooting

### Build Failures

**Issue**: "Failed to copy ../environment.py"
- **Solution**: Ensure you're deploying from the correct directory context
- The Dockerfile expects parent directory files

**Issue**: "Failed to install torch"
- **Solution**: Build takes time (3-5 min). If timeout, upgrade to paid plan
- Free tier has build time limits

### Runtime Issues

**Issue**: "Application crashed" or 503 errors
- **Solution**: Check logs for Python errors
- Verify model file `ppo_cart_1000000.zip` is included
- Check memory usage (Free tier: 512MB, Starter: 2GB)

**Issue**: WebSocket connection fails
- **Solution**: Ensure CORS is configured (already set in app.py)
- Check browser console for connection errors
- Verify eventlet is installed (already in requirements.txt)

**Issue**: Cold starts on Free tier
- **Solution**: Free tier spins down after 15 min of inactivity
- First request after spin-down takes 30-60 seconds
- Upgrade to Starter plan ($7/mo) for always-on service

### Performance Issues

**Issue**: Slow simulation rendering
- **Solution**: This is expected on Free tier (512MB RAM)
- Consider Starter plan (2GB RAM) for better performance
- Reduce simulation complexity if needed

## 🔒 Security Best Practices

1. **Secret Key**: Update Flask secret key in production
   ```python
   # In app.py, replace:
   app.config['SECRET_KEY'] = 'secret!'
   # With:
   app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret')
   ```
   Then add `SECRET_KEY` environment variable in Render settings

2. **CORS**: Currently set to `*` for development
   - For production, restrict to your domain:
   ```python
   socketio = SocketIO(app, cors_allowed_origins="https://yourdomain.com")
   ```

3. **Rate Limiting**: Consider adding rate limiting for production
   ```bash
   pip install Flask-Limiter
   ```

## 💰 Pricing

- **Free Tier**: 
  - 750 hours/month
  - 512MB RAM
  - Spins down after 15 min inactivity
  - Good for demos/testing

- **Starter Plan**: $7/month
  - Always-on
  - 2GB RAM
  - Better performance
  - Recommended for production

## 📚 Additional Resources

- [Render Documentation](https://render.com/docs)
- [Flask-SocketIO Documentation](https://flask-socketio.readthedocs.io/)
- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/configure.html)

## 🎉 Success!

Once deployed, your demo will be accessible at:
- `https://car-t-cell-demo.onrender.com` (or your custom domain)
- Share this URL with collaborators
- Monitor usage in Render Dashboard

## 🔄 Updates

To deploy updates:
1. Commit and push changes to your repository
2. Render auto-deploys (if enabled)
3. Or manually deploy from Render Dashboard

```bash
git add .
git commit -m "Update simulation parameters"
git push origin main
```

Render will automatically rebuild and redeploy your service.

