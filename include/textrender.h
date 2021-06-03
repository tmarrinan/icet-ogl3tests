#ifndef __TEXTRENDERER_H
#define __TEXTRENDERER_H

#include <cstdint>
#include <string>
#include <vector>
#include <ft2build.h>
#include FT_FREETYPE_H

static FT_Library ft;
static const unsigned long kOffsetsFromUtf8[4] = { 0x00000000UL, 0x00003080UL, 
                                                   0x000E2080UL, 0x03C82080UL };

typedef struct TR_FontFace
{
    FT_Face face;
    uint32_t size;
} TR_FontFace;

typedef struct TR_CharGlyph
{
    uint32_t charcode;
    uint32_t width;
    uint32_t height;
    uint8_t *pixels;
    int left;
    int top;
    int advance_x;
} TR_CharGlyph;

void TR_Initialize();
void TR_CreateFontFace(const char *font_file, uint32_t size, TR_FontFace **font_ptr);
void TR_DestroyFontFace(TR_FontFace *font);
void TR_SetFontSize(TR_FontFace *font, uint32_t size);
void TR_RenderStringToTexture(TR_FontFace *font, std::string utf8_text, bool flip_y,
                              uint32_t *width, uint32_t *height, uint32_t *baseline,
                              uint8_t **pixels);
void TR_RenderStringToTextureWithWrap(TR_FontFace *font, std::string utf8_text,
                                      uint32_t wrap_width, bool flip_y, uint32_t *width,
                                      uint32_t *height, uint32_t *baseline,
                                      uint8_t **pixels);

static void convertUtf8ToUtf32(unsigned char *source, uint16_t bytes, uint32_t *target);
static void getRenderedGlyphsFromString(TR_FontFace *font, std::string utf8_text,
                                        uint32_t *width, uint32_t *height, uint32_t *baseline,
                                        std::vector<TR_CharGlyph> *glyphs);

#endif // __TEXTRENDERER_H
