#include "textrender.h"

void TR_Initialize()
{
    if (FT_Init_FreeType(&ft))
    {
        fprintf(stderr, "Error: could not initialize freetype library\n");
    }
}

void TR_CreateFontFace(const char *font_file, uint32_t size, TR_FontFace **font_ptr)
{
    *font_ptr = (TR_FontFace*)malloc(sizeof(TR_FontFace));
    if (FT_New_Face(ft, font_file, 0, &((*font_ptr)->face)))
    {
        fprintf(stderr, "Error: could not open font\n");
        return;
    }
    (*font_ptr)->size = size;

    FT_Select_Charmap((*font_ptr)->face, FT_ENCODING_UNICODE);
    FT_Set_Pixel_Sizes((*font_ptr)->face, 0, (*font_ptr)->size);
}

void TR_DestroyFontFace(TR_FontFace *font)
{
    FT_Done_Face(font->face);
    free(font);
}

void TR_SetFontSize(TR_FontFace *font, uint32_t size)
{
    font->size = size;
    FT_Set_Pixel_Sizes(font->face, 0, font->size);
}

void TR_RenderStringToTexture(TR_FontFace *font, std::string utf8_text, bool flip_y,
                              uint32_t *width, uint32_t *height, uint32_t *baseline,
                              uint8_t **pixels)
{
    size_t i;
    uint32_t r, x, j, k, pt, ix, iy, idx;
    std::vector<TR_CharGlyph> glyphs;

    getRenderedGlyphsFromString(font, utf8_text, width, height, baseline, &glyphs);

    r = *width % 4;
    if (r != 0)
    {
        *width = *width + 4 - r;
    }
    r = *height % 4;
    if (r != 0)
    {
        *height = *height + 4 - r;
    }

    uint32_t size = (*width) * (*height);
    *pixels = (uint8_t*)malloc(size);
    memset(*pixels, 0, size);

    x = 0;
    for (i=0; i<glyphs.size(); i++)
    {
        pt = (*baseline) - glyphs[i].top;
        for (j=0; j<glyphs[i].height; j++)
        {
            iy = pt + j;
            if (flip_y)
            {
                iy = (*height) - iy - 1;
            }
            for (k=0; k<glyphs[i].width; k++)
            {
                ix = glyphs[i].left + x + k;
                idx = (*width) * iy + ix;
                if (idx < 0 || idx >= size)
                {
                    continue;
                }
                (*pixels)[idx] = glyphs[i].pixels[glyphs[i].width * j + k];
            }
        }
        x += glyphs[i].advance_x / 64;
        free(glyphs[i].pixels);
    }
    glyphs.clear();
}

void TR_RenderStringToTextureWithWrap(TR_FontFace *font, std::string utf8_text,
                                      uint32_t wrap_width, bool flip_y, uint32_t *width,
                                      uint32_t *height, uint32_t *baseline, uint8_t **pixels)
{
    int lines, line_height, space;
    size_t i;
    uint32_t r, x, j, k, pt, ix, iy, idx, total_w, total_h;
    bool hyphen = false;
    std::vector<TR_CharGlyph> glyphs;
    std::vector<size_t> line_breaks;

    getRenderedGlyphsFromString(font, utf8_text, &total_w, &total_h, baseline, &glyphs);

    x = 0;
    lines = 1;
    line_height = (int)((double)font->size * 1.333);
    space = -1;
    i = 0;
    while (i < glyphs.size())
    {
        if (x + glyphs[i].width > wrap_width)
        {
            x = 0;
            if (space >= 0)
            {
                if (hyphen)
                {
                    space++;
                }
                else
                {
                    free(glyphs[space].pixels);
                    glyphs.erase(glyphs.begin() + space);
                }
            }
            i = space >= 0 ? space : 1;
            line_breaks.push_back(i);
            lines++;
            space = -1;
            hyphen = false;
        }
        else
        {
            if (glyphs[i].charcode == ' ' || glyphs[i].charcode == '\t' || glyphs[i].charcode == '\n' || glyphs[i].charcode == '\r')
            {
                space = (int)i;
            }
            else if (glyphs[i].charcode == '-')
            {
                space = (int)i;
                hyphen = true;
            }
            x += glyphs[i].advance_x / 64;
            i++;
        }
    }
    line_breaks.push_back(glyphs.size());

    r = wrap_width % 4;
    if (r == 0)
    {
        *width = wrap_width;
    }
    else
    {
        *width = wrap_width + 4 - r;
    }
    r = (lines * line_height) % 4;
    if (r == 0)
    {
        *height = lines * line_height;
    }
    else
    {
        *height = (lines * line_height) + 4 - r;
    }

    uint32_t size = (*width) * (*height);
    *pixels = (uint8_t*)malloc(size);
    memset(*pixels, 0, size);

    lines = 0;
    x = 0;
    size_t l, start;
    start = 0;
    for (l=0; l<line_breaks.size(); l++)
    {
        x = 0;
        for (i=start; i<line_breaks[l]; i++)
        {
            pt = *baseline - glyphs[i].top + (lines * line_height);
            for (j=0; j<glyphs[i].height; j++)
            {
                iy = pt + j;
                if (flip_y)
                {
                    iy = (*height) - iy - 1;
                }
                for (k=0; k<glyphs[i].width; k++)
                {
                    ix = glyphs[i].left + x + k;
                    idx = (*width) * iy + ix;
                    if (idx < 0 || idx >= size)
                    {
                        continue;
                    }
                    (*pixels)[idx] = glyphs[i].pixels[glyphs[i].width * j + k];
                }
            }
            x += glyphs[i].advance_x / 64;
            free(glyphs[i].pixels);
        }
        start = line_breaks[l];
        lines++;
    }
}

//
// Internal use only
//
void convertUtf8ToUtf32(unsigned char *source, uint16_t bytes, uint32_t *target)
{
    int32_t ch = 0;

    switch (bytes) {
        case 4: ch += *source++; ch <<= 6;
        case 3: ch += *source++; ch <<= 6;
        case 2: ch += *source++; ch <<= 6;
        case 1: ch += *source++;
    }
    ch -= kOffsetsFromUtf8[bytes-1];

    *target = ch;
}

void getRenderedGlyphsFromString(TR_FontFace *font, std::string utf8_text, uint32_t *width,
                                 uint32_t *height, uint32_t *baseline,
                                 std::vector<TR_CharGlyph> *glyphs)
{
    int i = 0, top = 0, bottom = 0;
    *width = 0;
    unsigned char *unicode = (unsigned char*)utf8_text.c_str();
    do
    {
        uint32_t charcode = 0;
        if ((utf8_text[i] & 0x80) == 0)   // 1 byte
        {
            charcode = utf8_text[i];
            i++;
        }
        else if ((utf8_text[i] & 0xE0) == 0xC0)
        {
            convertUtf8ToUtf32(unicode + i, 2, &charcode);
            i += 2;
        }
        else if ((utf8_text[i] & 0xF0) == 0xE0)
        {
            convertUtf8ToUtf32(unicode + i, 3, &charcode);
            i += 3;
        }
        else if ((utf8_text[i] & 0xF8) == 0xF0)
        {
            convertUtf8ToUtf32(unicode + i, 4, &charcode);
            i += 4;
        }
        else
        {
            i++;
            continue;
        }

        uint32_t glyph_index = FT_Get_Char_Index(font->face, charcode);
        if (FT_Load_Glyph(font->face, glyph_index, FT_LOAD_RENDER))
        {
            continue;
        }

        glyphs->push_back(TR_CharGlyph());
        size_t c = glyphs->size() - 1;
        (*glyphs)[c].charcode = charcode;
        (*glyphs)[c].width = font->face->glyph->bitmap.width;
        (*glyphs)[c].height = font->face->glyph->bitmap.rows;
        uint32_t size = (*glyphs)[c].width * (*glyphs)[c].height;
        (*glyphs)[c].pixels = (uint8_t*)malloc(size);
        memcpy((*glyphs)[c].pixels, font->face->glyph->bitmap.buffer, size);
        (*glyphs)[c].left = font->face->glyph->bitmap_left;
        (*glyphs)[c].top = font->face->glyph->bitmap_top;
        (*glyphs)[c].advance_x = font->face->glyph->advance.x;

        *width += (*glyphs)[c].advance_x / 64;
        if ((*glyphs)[c].top > top)
        {
            top = (*glyphs)[c].top;
            *baseline = top;
        }
        if (((int)((*glyphs)[c].top) - (int)((*glyphs)[c].height)) < bottom)
        {
            bottom = (int)((*glyphs)[c].top) - (int)((*glyphs)[c].height);
        }
    } while (i < utf8_text.length());

    *height = top - bottom;
}