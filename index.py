import os
import base64
import json
import requests
from PIL import Image
import io

def encode_image(image_base64):
    """Process base64 image, resize to 512x512, and re-encode to base64."""
    # Decode base64 to image
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data))
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to 512x512 maintaining aspect ratio
    img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    
    # Create new image with padding if needed
    new_img = Image.new('RGB', (512, 512), (255, 255, 255))  # white background
    
    # Paste resized image in center
    offset = ((512 - img.size[0]) // 2, (512 - img.size[1]) // 2)
    new_img.paste(img, offset)
    
    # Save to bytes and re-encode to base64
    buffer = io.BytesIO()
    new_img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def main(req, res):
    """Analyze clothing items using GPT-4 Vision API."""
    # Get OpenAI API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return res.json({
            'error': 'OPENAI_API_KEY not found in environment variables'
        }, 500)
    
    # Get base64 images from request
    try:
        data = req.json
        if not data or 'images' not in data:
            return res.json({
                'error': 'No images provided in request body'
            }, 400)
        
        base64_images = data['images']
        if not isinstance(base64_images, list):
            base64_images = [base64_images]
        
        # Process and resize each image
        processed_images = [encode_image(img) for img in base64_images]
        
        system_prompt = """You are a fashion expert analyzing clothing items in images.
        For each item, identify:
        1. Type of clothing/accessory
        2. Brand (if visible or recognizable, otherwise suggest a similar brand that matches the style)
        3. Color (be specific with shades, or suggest closest matching color)
        4. Material (if visible, otherwise suggest most likely material based on appearance)
        5. Aesthetic/style (e.g. casual, formal, streetwear, etc.)
        6. Extra details (Include the specific clothing item if you know it, otherwise include a close alternative that fits the item. Be specific. Must be a real clothing item. i.e. "Converse Chuck Taylor All Star, Jordan 1, etc.")

        Keep responses concise. If you're unsure about any attribute provide educated suggestions based on visual cues and fashion knowledge.

        Format your response as a JSON array with one object per item. Each object should have these fields:
        {
            "type": "",
            "brand": "",
            "color": "",
            "material": "",
            "aesthetic": "",
            "extra_details": "",
            "confidence": 0.0
        }

        Return the results in the same order as the images provided. Never use 'unknown' - instead suggest similar alternatives."""
        
        # Create content with all images
        content = [
            {
                "type": "text",
                "text": f"Analyze these {len(processed_images)} clothing items. For EACH item, return a JSON object with these fields: type, brand, color, material, aesthetic, extra_details. Format your response as a JSON array with one object per item, in the same order as the images. Be specific but concise."
            }
        ]
        
        # Add each image to content
        for base64_image in processed_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        # Call GPT-4V API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        response_data = response.json()
        
        if 'error' in response_data:
            return res.json({
                'error': response_data['error']
            }, 500)
        
        # Extract and parse the response
        response_content = response_data['choices'][0]['message']['content']
        
        # Clean up markdown code blocks if present
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif "```" in response_content:
            response_content = response_content.split("```")[1].split("```")[0].strip()
        
        # Parse JSON response
        analyses = json.loads(response_content)
        if not isinstance(analyses, list):
            analyses = [analyses]
        
        # Add confidence scores
        for analysis in analyses:
            analysis["confidence"] = 0.9
        
        return res.json({
            'success': True,
            'analyses': analyses
        })
        
    except Exception as e:
        return res.json({
            'error': str(e)
        }, 500) 
